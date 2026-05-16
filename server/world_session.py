"""
InteractiveWorldSession — a stateful, steppable wrapper around WanI2VFast.

This refactors `WanI2VFast.generate()` (wan/image2video_fast.py) into a
persistent session that can be driven *live* by browser input:

    sess = InteractiveWorldSession(wan_i2v, cfg)
    sess.start(pil_image, prompt, size="480*832", seed=42)
    while playing:
        frames = sess.step(keys=["w", "l"])   # uint8 [N,H,W,3], N≈chunk_size*4

The model is causal with a (sliding-window) KV cache, and consumes camera
conditioning *per chunk*, so each `step()` extends the world by one chunk
while preserving long-term memory through the KV cache. Camera pose is
integrated statefully from WASD (move) + IJKL (look) key state, reusing the
exact motion model from wan/utils/wasd_ijkl_to_c2ws.py.

Faithfulness notes (documented approximations vs. the offline path):
  * VAE decode is per-chunk instead of one decode over the full clip. Minor
    boundary softness possible; mask=0 after frame 0 means the model relies
    on KV memory anyway.
  * Translation normalization in compute_relative_poses() is replaced by a
    fixed scale (TRANS_NORM) so motion magnitude is consistent across chunks
    instead of per-array max (which would be noisy chunk-to-chunk).
Everything inside the per-chunk diffusion loop is a line-for-line port of
generate(), so model-call semantics (kwargs, kv-cache update step) are identical.
"""

import math
import random
import sys

import numpy as np
import torch
import torch.nn.functional as NF
import torchvision.transforms.functional as TF
from einops import rearrange

from wan.configs import MAX_AREA_CONFIGS
from wan.utils.cam_utils import (
    SE3_inverse,
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
)
from wan.utils.wasd_ijkl_to_c2ws import get_rotation_matrix

# Per-latent-frame motion. One latent frame ~= 4 video frames, so these are
# ~4x the per-video-frame constants used in wasd_ijkl_to_c2ws.generate_and_save_trajectory.
MOVE_SPEED = 0.05 * 4
ROT_SPEED_RAD = np.deg2rad(2.0) * 4
PITCH_LIMIT = np.deg2rad(85)
# Fixed translation normalization (see module docstring). compute_relative_poses
# normalizes "to roughly 1 std"; a single full-speed step maps to ~1.0 here.
TRANS_NORM = MOVE_SPEED

# Default fast-inference intrinsics. The offline path loads intrinsics.npy from
# an example dir; for free-form play we use a reasonable pinhole for 480p
# (fx=fy≈ W*0.7, principal point centered) and let get_Ks_transformed rescale.
def _default_Ks(width_org=832, height_org=480):
    fx = fy = 0.70 * width_org
    cx, cy = width_org / 2.0, height_org / 2.0
    return torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32)


class InteractiveWorldSession:

    def __init__(self, wan_i2v, cfg):
        self.w = wan_i2v          # a constructed wan.WanI2VFast
        self.cfg = cfg
        self.device = wan_i2v.device
        self.active = False

    # ------------------------------------------------------------------ start
    @torch.no_grad()
    def start(
        self,
        img,                       # PIL.Image (RGB)
        prompt,
        size="480*832",
        seed=-1,
        chunk_size=3,
        timesteps_index=(0, 179, 358, 679),
        shift=3.0,                 # 480p recommendation from generate() docstring
        horizon_chunks=8,          # # of chunks for which the image/mask condition is built exactly
    ):
        w = self.w
        cfg = self.cfg
        self.chunk_size = chunk_size
        self.timesteps_index = list(timesteps_index)
        self.batch_size = 1
        max_area = MAX_AREA_CONFIGS[size]

        # ---- geometry (verbatim from generate()) ----
        img_t = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        ih, iw = img_t.shape[1:]
        aspect = ih / iw
        self.lat_h = round(np.sqrt(max_area * aspect) // w.vae_stride[1] //
                           w.patch_size[1] * w.patch_size[1])
        self.lat_w = round(np.sqrt(max_area / aspect) // w.vae_stride[2] //
                           w.patch_size[2] * w.patch_size[2])
        self.h = self.lat_h * w.vae_stride[1]
        self.wd = self.lat_w * w.vae_stride[2]

        self.frame_seqlen = int((self.lat_h * self.lat_w) // 4)
        self.max_seq_len = int(math.ceil(
            chunk_size * self.lat_h * self.lat_w /
            (w.patch_size[1] * w.patch_size[2]) / w.sp_size)) * w.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        self.seed_g = torch.Generator(device=self.device)
        self.seed_g.manual_seed(seed)

        # ---- text context (verbatim) ----
        if not w.t5_cpu:
            w.text_encoder.model.to(self.device)
            self.context = w.text_encoder([prompt], self.device)
        else:
            self.context = [t.to(self.device)
                            for t in w.text_encoder([prompt], torch.device('cpu'))]

        # ---- scheduler / timesteps (verbatim) ----
        w.scheduler.set_timesteps(w.num_train_timesteps, shift=shift)
        self.timesteps = w.scheduler.timesteps[self.timesteps_index]

        # ---- image+mask condition for the first `horizon_chunks` chunks ----
        # Built exactly like generate(): frame-0 mask = 1, rest 0; VAE-encode
        # [image, zeros...]. Beyond the horizon we append zero-condition chunks
        # (msk=0, VAE(zeros)) which is exactly what generate() feeds chunks>0.
        lat_f = chunk_size * horizon_chunks
        F = (lat_f - 1) * 4 + 1
        msk = torch.ones(1, F, self.lat_h, self.lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
            dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, self.lat_h, self.lat_w)
        msk = msk.transpose(1, 2)[0]                          # [4, lat_f, lh, lw]

        vid = torch.concat([
            NF.interpolate(img_t[None].cpu(), size=(self.h, self.wd),
                           mode='bicubic').transpose(0, 1),
            torch.zeros(3, F - 1, self.h, self.wd)
        ], dim=1).to(self.device)
        y = w.vae.encode([vid])[0]                            # [16, lat_f, lh, lw]
        y = torch.concat([msk, y])                            # [20, lat_f, lh, lw]
        self._cond_chunks = list(y.split(chunk_size, dim=1))
        self._zero_video_frames = (chunk_size - 1) * 4 + 1     # for zero extensions

        # ---- KV caches (verbatim) ----
        m = w.model.config
        tdt = w.pipe_dtype
        if w.local_attn_size > -1:
            kv_size = self.frame_seqlen * w.local_attn_size
        else:
            # unbounded session → cap at the horizon to keep VRAM bounded.
            kv_size = self.frame_seqlen * lat_f
        head_dim = m.dim // m.num_heads
        local_heads = m.num_heads // w.sp_size
        self.self_kv = w._initialize_self_kv_cache(
            m.num_layers, [self.batch_size, kv_size, local_heads, head_dim],
            tdt, self.device)
        self.cross_kv = w._initialize_crossattn_cache(
            m.num_layers,
            [self.batch_size, 512, m.num_heads, head_dim], tdt, self.device)
        self.kv_size = kv_size

        # ---- camera state ----
        self._c2w = np.eye(4)
        self._prev_c2w = np.eye(4)        # anchor for framewise-relative deltas
        self._pitch = 0.0
        self._Ks = get_Ks_transformed(
            _default_Ks(), height_org=480, width_org=832,
            height_resize=self.h, width_resize=self.wd,
            height_final=self.h, width_final=self.wd)[0].to(self.device)

        self.chunk_id = 0
        self.active = True
        self._horizon_chunks = horizon_chunks

    # ----------------------------------------------------- camera integration
    def _advance_camera(self, keys):
        """Advance the camera one *latent* frame given held keys (WASD+IJKL)."""
        keys = set(k.lower() for k in keys)
        R = self._c2w[:3, :3]
        T = self._c2w[:3, 3]

        pitch_delta = 0.0
        if 'i' in keys:
            pitch_delta += ROT_SPEED_RAD
        if 'k' in keys:
            pitch_delta -= ROT_SPEED_RAD
        if not (-PITCH_LIMIT <= self._pitch + pitch_delta <= PITCH_LIMIT):
            pitch_delta = 0.0
        self._pitch += pitch_delta
        R_pitch = get_rotation_matrix('x', pitch_delta)

        yaw_delta = 0.0
        if 'j' in keys:
            yaw_delta -= ROT_SPEED_RAD
        if 'l' in keys:
            yaw_delta += ROT_SPEED_RAD
        R_yaw = get_rotation_matrix('y', yaw_delta)

        R_new = R_yaw @ R @ R_pitch
        fwd = R_new[:, 2]
        right = R_new[:, 0]
        fwd_flat = np.array([fwd[0], 0, fwd[2]])
        right_flat = np.array([right[0], 0, right[2]])
        fn = np.linalg.norm(fwd_flat)
        rn = np.linalg.norm(right_flat)
        fwd_flat = fwd_flat / (fn + 1e-6) if fn > 0 else fwd_flat
        right_flat = right_flat / (rn + 1e-6) if rn > 0 else right_flat

        mv = np.zeros(3)
        if 'w' in keys:
            mv += fwd_flat * MOVE_SPEED
        if 's' in keys:
            mv -= fwd_flat * MOVE_SPEED
        if 'd' in keys:
            mv += right_flat * MOVE_SPEED
        if 'a' in keys:
            mv -= right_flat * MOVE_SPEED

        c2w = np.eye(4)
        c2w[:3, :3] = R_new
        c2w[:3, 3] = T + mv
        self._c2w = c2w
        return c2w.copy()

    def _chunk_plucker(self, keys_per_frame):
        """Build [b, 6, chunk, lat_h, lat_w] plucker emb for the next chunk.

        Uses framewise-relative poses anchored on the previous chunk's last
        pose, with a *fixed* translation scale (TRANS_NORM) for cross-chunk
        consistency (see module docstring).
        """
        poses = [self._prev_c2w.copy()]
        for keys in keys_per_frame:
            poses.append(self._advance_camera(keys))
        self._prev_c2w = poses[-1].copy()

        c2ws = torch.from_numpy(np.stack(poses)).float().to(self.device)  # [n+1,4,4]
        # framewise relative, but with fixed-scale translation normalization
        ref_w2c = SE3_inverse(c2ws[0:1])
        rel = torch.matmul(ref_w2c, c2ws)
        rel[0] = torch.eye(4, device=c2ws.device, dtype=c2ws.dtype)
        rel_fw = torch.bmm(SE3_inverse(rel[:-1]), rel[1:])
        rel[1:] = rel_fw
        rel[:, :3, 3] = rel[:, :3, 3] / TRANS_NORM
        rel = rel[1:]                                          # drop anchor → [chunk,4,4]

        Ks = self._Ks.repeat(rel.shape[0], 1)
        emb = get_plucker_embeddings(rel, Ks, self.h, self.wd, only_rays_d=False)
        emb = rearrange(
            emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(self.h // self.lat_h), c2=int(self.wd // self.lat_w))[None]
        emb = rearrange(emb, 'b (f h w) c -> b c f h w',
                        f=rel.shape[0], h=self.lat_h, w=self.lat_w)
        return emb.to(self.w.param_dtype)

    def _cond_for_chunk(self):
        if self.chunk_id < len(self._cond_chunks):
            return self._cond_chunks[self.chunk_id]
        # Beyond precomputed horizon: zero condition (matches generate() chunks>0).
        msk0 = torch.zeros(4, self.chunk_size, self.lat_h, self.lat_w,
                           device=self.device)
        zv = torch.zeros(3, self._zero_video_frames, self.h, self.wd,
                         device=self.device)
        zlat = self.w.vae.encode([zv])[0]                      # [16, chunk, lh, lw]
        return torch.concat([msk0, zlat])

    # ------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, keys):
        """Generate the next chunk. `keys` = list of held key chars (applied to
        every frame of the chunk). Returns uint8 ndarray [N, H, W, 3]."""
        assert self.active, "call start() first"
        w = self.w
        cs = self.chunk_size
        keys_per_frame = [list(keys)] * cs

        noise = torch.randn(16, cs, self.lat_h, self.lat_w,
                            dtype=torch.float32, generator=self.seed_g,
                            device=self.device)
        cond = self._cond_for_chunk()
        plucker = self._chunk_plucker(keys_per_frame)

        kwargs = {
            'context': [self.context[0]],
            'seq_len': self.max_seq_len,
            'y': [cond],
            'dit_cond_dict': {'c2ws_plucker_emb': plucker.chunk(1, dim=0)},
            'kv_cache': self.self_kv,
            'crossattn_cache': self.cross_kv,
            'current_start': self.chunk_id * cs * self.frame_seqlen,
            'max_attention_size': self.kv_size,
        }

        with torch.amp.autocast('cuda', dtype=w.param_dtype), torch.no_grad():
            cur = noise
            x0 = None
            for ti in range(len(self.timesteps)):
                t = torch.stack([self.timesteps[ti]]).to(self.device)
                noise_pred = w.model(x=[cur.to(self.device)], t=t, **kwargs)[0]
                x0 = w._convert_flow_pred_to_x0(
                    flow_pred=noise_pred, xt=cur,
                    timestep=self.timesteps[ti], scheduler=w.scheduler)
                if ti < len(self.timesteps) - 1:
                    nt = self.timesteps[ti + 1]
                    cur = w.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=self.seed_g,
                                        device=x0.device, dtype=x0.dtype), nt)
                else:
                    break

            # KV-cache update step (verbatim from generate())
            t0 = torch.stack([self.timesteps[-1] * 0.0]).to(self.device)
            w.model(x=[x0], t=t0, **kwargs)

            video = w.vae.decode([x0])[0]      # [3, n, H, W] in [-1,1]

        self.chunk_id += 1

        v = video.float().clamp(-1, 1).add(1).div(2)           # → [0,1]
        v = (v * 255).round().byte().cpu().numpy()             # [3, n, H, W]
        v = np.transpose(v, (1, 2, 3, 0))                      # [n, H, W, 3]
        return v

    def stop(self):
        self.active = False
        for c in (getattr(self, 'self_kv', None) or []):
            c.clear()
        for c in (getattr(self, 'cross_kv', None) or []):
            c.clear()
        torch.cuda.empty_cache()
