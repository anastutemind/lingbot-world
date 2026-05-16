"""
LingBot-World interactive browser server.

FastAPI + WebSocket. Loads WanI2VFast once at startup and serves a single
live "playable world" session: the browser creates an environment (image +
prompt), then streams held WASD/IJKL key state; a background worker keeps
generating chunks with the latest key state and streams JPEG frames back at
a steady playback rate (client absorbs jitter).

One session at a time (the model is a single un-batched instance). Concurrent
players would need model replicas / request batching — out of scope for v1.

Env vars:
  LBW_CKPT_DIR   checkpoint dir (default: lingbot-world-base-cam)
  LBW_TASK       wan task (default: i2v-A14B)
  LBW_SIZE       e.g. 480*832 (default) | 720*1280
  LBW_T5_CPU     "1" to keep T5 on CPU (saves VRAM on single GPU)
  LBW_PLAY_FPS   playback fps the sender targets (default 16)
  LBW_PORT       http port (default 8000)
"""

import asyncio
import base64
import io
import logging
import os
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS

from world_session import InteractiveWorldSession

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("lbw")

CKPT_DIR = os.environ.get("LBW_CKPT_DIR", "lingbot-world-base-cam")
TASK = os.environ.get("LBW_TASK", "i2v-A14B")
SIZE = os.environ.get("LBW_SIZE", "480*832")
T5_CPU = os.environ.get("LBW_T5_CPU", "1") == "1"
PLAY_FPS = float(os.environ.get("LBW_PLAY_FPS", "16"))
HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="LingBot-World Interactive")

# ---- global model (loaded once) ----
MODEL = {"wan": None, "cfg": None, "ready": False, "error": None}
SESSION_LOCK = asyncio.Lock()       # only one live session at a time


def _load_model():
    cfg = WAN_CONFIGS[TASK]
    log.info("Loading WanI2VFast from %s (t5_cpu=%s) …", CKPT_DIR, T5_CPU)
    t0 = time.time()
    wan_i2v = wan.WanI2VFast(
        config=cfg,
        checkpoint_dir=CKPT_DIR,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=T5_CPU,
        convert_model_dtype=True,
        local_attn_size=-1,
        sink_size=0,
    )
    MODEL["wan"] = wan_i2v
    MODEL["cfg"] = cfg
    MODEL["ready"] = True
    log.info("Model ready in %.1fs", time.time() - t0)


@app.on_event("startup")
def _startup():
    threading.Thread(target=_guarded_load, daemon=True).start()


def _guarded_load():
    try:
        _load_model()
    except Exception as e:                       # surface load failures to UI
        log.exception("model load failed")
        MODEL["error"] = repr(e)


@app.get("/healthz")
def healthz():
    return {"ready": MODEL["ready"], "error": MODEL["error"],
            "size": SIZE, "ckpt": CKPT_DIR}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/examples")
def examples():
    """List built-in starter images shipped in the repo's examples/ dir."""
    root = os.path.join(HERE, "..", "examples")
    out = []
    if os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d, "image.jpg")
            if os.path.isfile(p):
                out.append(d)
    return {"examples": out}


@app.get("/examples/{name}")
def example_image(name: str):
    name = os.path.basename(name)
    p = os.path.join(HERE, "..", "examples", name, "image.jpg")
    if not os.path.isfile(p):
        return FileResponse(os.path.join(HERE, "static", "index.html"),
                            status_code=404)
    return FileResponse(p, media_type="image/jpeg")


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")),
          name="static")


def _decode_image(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _jpeg(frame_rgb: np.ndarray, q=82) -> bytes:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else b""


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()

    if not MODEL["ready"]:
        await sock.send_json({"type": "error",
                              "msg": "model still loading" if not MODEL["error"]
                              else f"model load failed: {MODEL['error']}"})
        await sock.close()
        return

    if SESSION_LOCK.locked():
        await sock.send_json({"type": "busy",
                              "msg": "another session is active — try again shortly"})
        await sock.close()
        return

    async with SESSION_LOCK:
        await _run_session(sock)


async def _run_session(sock: WebSocket):
    loop = asyncio.get_running_loop()
    sess = InteractiveWorldSession(MODEL["wan"], MODEL["cfg"])
    state = {"keys": [], "playing": False, "stop": False}
    frame_q: asyncio.Queue = asyncio.Queue(maxsize=64)

    # ---- start handshake ----
    try:
        msg = await sock.receive_json()
    except (WebSocketDisconnect, RuntimeError):
        return
    if msg.get("type") != "start":
        await sock.send_json({"type": "error", "msg": "expected 'start'"})
        return

    try:
        img = _decode_image(msg["image"])
        prompt = msg.get("prompt") or "a vivid, explorable world"
        await sock.send_json({"type": "status", "msg": "initializing world…"})
        await loop.run_in_executor(
            None, lambda: sess.start(img, prompt, size=SIZE,
                                     seed=int(msg.get("seed", 42))))
    except Exception as e:
        log.exception("start failed")
        await sock.send_json({"type": "error", "msg": f"start failed: {e!r}"})
        return

    state["playing"] = True
    await sock.send_json({"type": "ready", "fps": PLAY_FPS})

    # ---- generation worker: keep producing chunks with the latest key state ----
    def gen_blocking():
        return sess.step(list(state["keys"]))

    async def producer():
        try:
            n = 0
            while not state["stop"]:
                t0 = time.time()
                frames = await loop.run_in_executor(None, gen_blocking)
                n += 1
                log.info("chunk %d: %d frames in %.1fs (keys=%s)",
                         n, len(frames), time.time() - t0, state["keys"])
                for f in frames:
                    if state["stop"]:
                        break
                    await frame_q.put(f)
        except Exception as e:
            log.exception("producer error")
            await frame_q.put(("__err__", repr(e)))

    # ---- sender: steady-rate playback drain (client also buffers) ----
    async def sender():
        period = 1.0 / PLAY_FPS
        nxt = time.time()
        sent = 0
        t_fps = time.time()
        while not state["stop"]:
            try:
                item = await asyncio.wait_for(frame_q.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(item, tuple) and item[0] == "__err__":
                await sock.send_json({"type": "error", "msg": item[1]})
                state["stop"] = True
                break
            await sock.send_bytes(_jpeg(item))
            sent += 1
            if time.time() - t_fps >= 2.0:
                await sock.send_json({"type": "stat",
                                      "gen_fps": round(sent / (time.time() - t_fps), 1),
                                      "qlen": frame_q.qsize()})
                sent = 0
                t_fps = time.time()
            nxt += period
            sleep = nxt - time.time()
            if sleep > 0:
                await asyncio.sleep(sleep)
            else:
                nxt = time.time()        # fell behind → reset cadence

    # ---- receiver: live key-state updates ----
    async def receiver():
        while not state["stop"]:
            try:
                m = await sock.receive_json()
            except (WebSocketDisconnect, RuntimeError):
                state["stop"] = True
                break
            t = m.get("type")
            if t == "action":
                state["keys"] = [str(k).lower() for k in m.get("keys", [])][:8]
            elif t == "stop":
                state["stop"] = True
                break

    try:
        await asyncio.gather(producer(), sender(), receiver())
    finally:
        state["stop"] = True
        await loop.run_in_executor(None, sess.stop)
        try:
            await sock.close()
        except RuntimeError:
            pass
        log.info("session ended")
