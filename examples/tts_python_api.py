#!/usr/bin/env python3
"""Minimal TTS demo — Python API, web UI with streaming audio.

Serves a single-page UI at http://localhost:8080.
Type text, pick a voice, adjust the pause slider, click Synthesise — audio
streams directly to the browser as it is generated.

Usage:
    python examples/tts_python_api.py
    python examples/tts_python_api.py --port 9090 --voice-dir /path/to/voices
"""

import argparse
import base64
import json
import queue
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_SENTENCE_END_RE = re.compile(r'[.!?]\s*$')


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, keeping terminal punctuation with each fragment."""
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voxtral TTS</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee;
         display: flex; flex-direction: column; align-items: center;
         justify-content: center; min-height: 100vh; gap: 1.2rem; padding: 2rem; }
  h1 { color: #e94560; font-size: 1.4rem; }
  .card { background: #16213e; border: 1px solid #333; border-radius: 12px;
          padding: 1.5rem; width: 100%; max-width: 600px; display: flex;
          flex-direction: column; gap: 0.9rem; }
  label { font-size: 0.8rem; color: #aaa; }
  textarea { width: 100%; padding: 0.6rem 0.75rem; border: 1px solid #333;
             border-radius: 8px; background: #0f3460; color: #eee;
             font-size: 0.95rem; resize: vertical; min-height: 100px; outline: none; }
  textarea:focus { border-color: #e94560; }
  .row { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }
  select { padding: 0.4rem 0.6rem; border: 1px solid #333; border-radius: 6px;
           background: #0f3460; color: #eee; font-size: 0.85rem; }
  input[type=range] { width: 100px; accent-color: #e94560; cursor: pointer; }
  .val { font-size: 0.82rem; color: #eee; min-width: 3.5em; }
  button { padding: 0.55rem 1.4rem; border: none; border-radius: 8px; cursor: pointer;
           font-size: 0.95rem; font-weight: 600; background: #e94560; color: #fff; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  #status { font-size: 0.8rem; color: #888; min-height: 1.2em; }
</style>
</head>
<body>
<h1>Voxtral TTS</h1>
<div class="card">
  <label>Text</label>
  <textarea id="text" placeholder="Enter text to synthesise..."></textarea>
  <div class="row">
    <label>Voice:</label>
    <select id="voice"><option>neutral_female</option></select>
    <label>Intersentence Pause:</label>
    <input type="range" id="pause-slider" min="0" max="1000" step="50" value="0">
    <span class="val" id="pause-val">0 ms</span>
  </div>
  <div class="row">
    <button id="btn" onclick="synthesise()">Synthesise</button>
    <button onclick="stopAudio()" style="background:#333">Stop</button>
  </div>
  <div id="status"></div>
</div>

<script>
// ── Audio streaming (PCM s16le, 24 kHz, mono) ──────────────────────────
const SAMPLE_RATE = 24000;
let audioCtx = null;
let nextPlayTime = 0;

function ensureAudioCtx() {
  if (!audioCtx || audioCtx.state === 'closed') {
    audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  }
  if (audioCtx.state === 'suspended') audioCtx.resume();
}

function stopAudio() {
  if (audioCtx && audioCtx.state !== 'closed') {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }
  nextPlayTime = 0;
  if (abortCtrl) { abortCtrl.abort(); abortCtrl = null; }
  document.getElementById('btn').disabled = false;
  document.getElementById('status').textContent = '';
}

function enqueueAudio(base64pcm) {
  ensureAudioCtx();
  const raw = atob(base64pcm);
  const samples = raw.length / 2;
  if (samples === 0) return;
  const buf = audioCtx.createBuffer(1, samples, SAMPLE_RATE);
  const chan = buf.getChannelData(0);
  for (let i = 0; i < samples; i++) {
    let s = raw.charCodeAt(i * 2) | (raw.charCodeAt(i * 2 + 1) << 8);
    if (s >= 0x8000) s -= 0x10000;
    chan[i] = s / 32768;
  }
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  const now = audioCtx.currentTime;
  if (nextPlayTime < now) nextPlayTime = now;
  src.start(nextPlayTime);
  nextPlayTime += buf.duration;
}

// ── Pause control ─────────────────────────────────────────────────────
const slider = document.getElementById('pause-slider');
const valEl = document.getElementById('pause-val');
slider.addEventListener('input', async () => {
  valEl.textContent = slider.value + ' ms';
  await fetch('/pause', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pause_ms: parseInt(slider.value) }),
  }).catch(() => {});
});

async function loadPause() {
  try {
    const r = await fetch('/pause');
    const d = await r.json();
    slider.value = d.pause_ms;
    valEl.textContent = d.pause_ms + ' ms';
  } catch(e) {}
}
loadPause();

// ── Voices ────────────────────────────────────────────────────────────
async function loadVoices() {
  try {
    const r = await fetch('/voices');
    const d = await r.json();
    const sel = document.getElementById('voice');
    const prev = sel.value;
    sel.innerHTML = '';
    for (const v of d.voices) {
      const o = document.createElement('option'); o.value = v; o.textContent = v;
      if (v === prev) o.selected = true;
      sel.appendChild(o);
    }
  } catch(e) {}
}
loadVoices();

// ── Synthesis ─────────────────────────────────────────────────────────
let abortCtrl = null;

async function synthesise() {
  const text = document.getElementById('text').value.trim();
  if (!text) return;

  stopAudio();
  abortCtrl = new AbortController();

  const voice = document.getElementById('voice').value;
  const btn = document.getElementById('btn');
  const status = document.getElementById('status');

  btn.disabled = true;
  status.textContent = 'Synthesising...';

  try {
    const resp = await fetch('/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice }),
      signal: abortCtrl.signal,
    });
    if (!resp.ok) { status.textContent = 'Error: ' + resp.statusText; return; }

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\\n\\n')) !== -1) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let evType = 'message', evData = '';
        for (const line of block.split('\\n')) {
          if (line.startsWith('event: ')) evType = line.slice(7);
          else if (line.startsWith('data: ')) evData = line.slice(6);
        }
        if (evType === 'audio') enqueueAudio(evData);
        else if (evType === 'done') status.textContent = '';
        else if (evType === 'error') status.textContent = 'Error: ' + evData;
      }
    }
  } catch(e) {
    if (e.name !== 'AbortError') status.textContent = 'Error: ' + e.message;
  } finally {
    abortCtrl = null;
    btn.disabled = false;
  }
}

document.getElementById('text').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) synthesise();
});
</script>
</body>
</html>
"""


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/voices":
            data = json.dumps({"voices": _tts.list_voices()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/pause":
            data = json.dumps({"pause_ms": _pause_ms}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/stream":
            self._handle_stream()
        elif self.path == "/pause":
            self._handle_pause_post()
        else:
            self.send_error(404)

    def _handle_pause_post(self):
        global _pause_ms
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        _pause_ms = max(0, min(int(body.get("pause_ms", _pause_ms)), 1000))
        data = json.dumps({"pause_ms": _pause_ms}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_stream(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        text = body.get("text", "").strip()
        raw_voice = body.get("voice", None)
        voice = (raw_voice.strip() if isinstance(raw_voice, str) else None) or "neutral_female"

        if not text:
            self.send_error(400, "No text")
            return
        if len(text) > 4096:
            self.send_error(400, "Text too long")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        import numpy as np

        q: queue.Queue = queue.Queue()
        cancel = threading.Event()

        def gen():
            try:
                for sentence in _split_sentences(text):
                    if cancel.is_set():
                        break
                    pause = _pause_ms if _SENTENCE_END_RE.search(sentence) else 0
                    for chunk in _tts.stream(sentence, voice=voice, trailing_silence_ms=pause, sanitize_text=_sanitize_text):
                        if cancel.is_set():
                            break
                        pcm = (chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                        q.put(("audio", base64.b64encode(pcm).decode("ascii")))
            except Exception as e:
                q.put(("error", str(e)))
            finally:
                q.put(("done", ""))

        threading.Thread(target=gen, daemon=True).start()

        try:
            while True:
                ev_type, ev_data = q.get()
                self.wfile.write(f"event: {ev_type}\ndata: {ev_data}\n\n".encode())
                self.wfile.flush()
                if ev_type in ("done", "error"):
                    break
        except BrokenPipeError:
            cancel.set()

    def log_message(self, fmt, *args):
        pass


def main():
    global _tts, _pause_ms, _sanitize_text

    parser = argparse.ArgumentParser(description="Voxtral TTS web demo — Python API")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--voice-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantize", default="nf4")
    parser.add_argument("--pause-ms", type=int, default=0,
                        help="Default trailing silence in ms (default: 0)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--no-sanitize-text", action="store_true",
        help="Disable Latin-only character filter to support Arabic, Hindi, and other non-Latin scripts",
    )
    args = parser.parse_args()

    _pause_ms = args.pause_ms
    _sanitize_text = not args.no_sanitize_text

    from voxtral import VoxtralTTS

    print("Loading VoxtralTTS...")
    _tts = VoxtralTTS(
        device=args.device,
        quantize=args.quantize,
        custom_voice_dir=args.voice_dir,
        use_cache=not args.no_cache,
    )
    _tts.wait_for_ready()
    print(f"Ready. Voices: {', '.join(_tts.list_voices())}")
    print(f"Serving at http://{args.host}:{args.port}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


_tts = None
_pause_ms: int = 0
_sanitize_text: bool = True

if __name__ == "__main__":
    main()
