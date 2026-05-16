"use strict";
// LingBot-World interactive client.
// - creates an environment (image + prompt) and opens a WS session
// - receives JPEG frames, buffers them (jitter absorption), plays at server fps
// - streams held WASD/IJKL key state continuously to the server

const $ = (id) => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");

let imgDataURL = null;
let ws = null;
let serverFps = 16;

// ---------- server health ----------
async function poll() {
  try {
    const h = await (await fetch("/healthz")).json();
    const d = $("srvdot");
    if (h.error) { d.className = "dot bad"; $("srvtxt").textContent = "model load failed"; }
    else if (h.ready) { d.className = "dot ok"; $("srvtxt").textContent = `model ready · ${h.size} · ${h.ckpt}`; }
    else { d.className = "dot"; $("srvtxt").textContent = "model loading… (first start is slow)"; }
  } catch { $("srvtxt").textContent = "server unreachable"; }
}
poll(); setInterval(poll, 4000);

// ---------- starter samples ----------
fetch("/examples").then(r => r.json()).then(d => {
  const box = $("samples");
  (d.examples || []).forEach(name => {
    const b = document.createElement("button");
    b.textContent = "sample " + name;
    b.onclick = async () => {
      const blob = await (await fetch("/examples/" + name)).blob();
      setImage(await blobToDataURL(blob));
    };
    box.appendChild(b);
  });
}).catch(() => {});

const blobToDataURL = (b) => new Promise(res => {
  const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(b);
});

function setImage(durl) {
  imgDataURL = durl;
  const t = $("thumb"); t.src = durl; t.style.display = "block";
  $("go").disabled = false;
}

// ---------- image picking ----------
const drop = $("drop"), file = $("file");
drop.onclick = () => file.click();
file.onchange = () => file.files[0] && blobToDataURL(file.files[0]).then(setImage);
["dragover", "dragenter"].forEach(e =>
  drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("hot"); }));
["dragleave", "drop"].forEach(e =>
  drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("hot"); }));
drop.addEventListener("drop", ev => {
  const f = ev.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) blobToDataURL(f).then(setImage);
});

// ---------- jitter buffer + playback ----------
const buf = [];           // decoded ImageBitmaps waiting to play
let playing = false, prebuffered = false;
const PREBUFFER = 12;     // frames to accumulate before playback starts
let lastDraw = 0, pf = 0, pfT = performance.now();

function playLoop(ts) {
  if (!playing) return;
  requestAnimationFrame(playLoop);
  const period = 1000 / serverFps;
  if (ts - lastDraw < period) return;
  if (!prebuffered) {
    if (buf.length >= PREBUFFER) prebuffered = true; else { hud(); return; }
  }
  const frame = buf.shift();
  if (frame) {
    ctx.drawImage(frame, 0, 0, cv.width, cv.height);
    frame.close && frame.close();
    lastDraw = ts; pf++;
    if (buf.length === 0) prebuffered = false;   // underrun → re-prebuffer
  }
  if (performance.now() - pfT >= 1000) {
    $("hplay").textContent = pf; pf = 0; pfT = performance.now();
  }
  hud();
}
function hud() {
  $("hbuf").textContent = buf.length;
}

// ---------- session ----------
$("go").onclick = () => {
  if (!imgDataURL) return;
  $("go").disabled = true;
  $("hstate").textContent = "connecting";
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "blob";

  ws.onopen = () => {
    ws.send(JSON.stringify({
      type: "start", image: imgDataURL,
      prompt: $("prompt").value, seed: parseInt($("seed").value || "42", 10),
    }));
    $("hstate").textContent = "starting";
  };

  ws.onmessage = async (ev) => {
    if (typeof ev.data !== "string") {           // binary = JPEG frame
      const bmp = await createImageBitmap(ev.data);
      if (buf.length < 240) buf.push(bmp); else bmp.close && bmp.close();
      return;
    }
    const m = JSON.parse(ev.data);
    if (m.type === "ready") {
      serverFps = m.fps || 16;
      $("overlay").classList.add("hide");
      $("hstate").textContent = "live";
      playing = true; prebuffered = false; requestAnimationFrame(playLoop);
      startControls();
    } else if (m.type === "status") {
      $("hmsg").textContent = m.msg;
    } else if (m.type === "stat") {
      $("hgen").textContent = m.gen_fps;
    } else if (m.type === "busy") {
      alert("Server busy: " + m.msg); endSession("busy");
    } else if (m.type === "error") {
      alert("Error: " + m.msg); endSession("error: " + m.msg);
    }
  };
  ws.onclose = () => endSession("disconnected");
  ws.onerror = () => endSession("ws error");
};

function endSession(why) {
  playing = false; prebuffered = false; buf.length = 0;
  $("hstate").textContent = why || "idle";
  $("go").disabled = !imgDataURL;
  $("overlay").classList.remove("hide");
  stopControls();
}

// ---------- controls ----------
const held = new Set();
const MAP = { arrowup: "i", arrowdown: "k", arrowleft: "j", arrowright: "l" };
const VALID = new Set(["w", "a", "s", "d", "i", "j", "k", "l"]);
let hbInterval = null, dirty = false;

function sendKeys() {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: "action", keys: [...held] }));
    dirty = false;
  }
}
function onKey(down) {
  return (e) => {
    let k = e.key.toLowerCase();
    if (MAP[k]) k = MAP[k];
    if (!VALID.has(k)) {
      if (e.key === "Escape") { held.clear(); dirty = true; }
      return;
    }
    e.preventDefault();
    if (down) held.add(k); else held.delete(k);
    dirty = true;
  };
}
const kd = onKey(true), ku = onKey(false);
function startControls() {
  window.addEventListener("keydown", kd);
  window.addEventListener("keyup", ku);
  // heartbeat: push current key state ~6x/s (and immediately on change)
  hbInterval = setInterval(() => { if (dirty) sendKeys(); }, 160);
}
function stopControls() {
  window.removeEventListener("keydown", kd);
  window.removeEventListener("keyup", ku);
  if (hbInterval) clearInterval(hbInterval);
  held.clear();
}
cv.addEventListener("click", () => cv.focus());
