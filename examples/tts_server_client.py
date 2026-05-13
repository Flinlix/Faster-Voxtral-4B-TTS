#!/usr/bin/env python3
"""Minimal TTS demo — server client, web UI with streaming audio.

Connects to a running Voxtral server (server.py) and serves a single-page UI
at http://localhost:8081. Audio streams directly to the browser as it arrives
from the TTS server.

Start the TTS server first:
    python server.py --port 8000

Then run this client:
    python examples/tts_server_client.py --tts-url http://localhost:8000
    python examples/tts_server_client.py --port 9090 --tts-url http://remote-host:8000
"""

import argparse
import base64
import http.client
import json
import re
import struct
import threading
import urllib.error
import urllib.parse
import urllib.request
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

# WAV header size to skip when the TTS server returns response_format=wav
_WAV_HEADER_SIZE = 44


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
            self._proxy_voices()
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

    def _proxy_voices(self):
        url = f"{_tts_url}/v1/voices"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = resp.read()
        except Exception as e:
            data = json.dumps({"voices": [], "error": str(e)}).encode()
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

        cancel = threading.Event()

        def stream_from_server():
            """Stream PCM sentence-by-sentence from TTS server, forward as SSE audio events."""
            sentences = _split_sentences(text)

            parsed = urllib.parse.urlparse(_tts_url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)

            for sentence in sentences:
                if cancel.is_set():
                    break
                pause = _pause_ms if _SENTENCE_END_RE.search(sentence) else 0

                payload = json.dumps({
                    "model": "voxtral-4b",
                    "input": sentence,
                    "voice": voice,
                    "response_format": "pcm",
                    "trailing_silence_ms": pause,
                }).encode()

                if parsed.scheme == "https":
                    import ssl
                    conn = http.client.HTTPSConnection(host, port, timeout=120,
                                                       context=ssl.create_default_context())
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=120)

                try:
                    conn.request("POST", "/v1/audio/speech", body=payload,
                                 headers={"Content-Type": "application/json"})
                    resp = conn.getresponse()

                    if resp.status != 200:
                        err = resp.read().decode(errors="replace")
                        self.wfile.write(
                            f"event: error\ndata: HTTP {resp.status}: {err}\n\n".encode())
                        self.wfile.flush()
                        return

                    chunk_size = 4096  # bytes ≈ 85 ms at 24 kHz
                    while not cancel.is_set():
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        b64 = base64.b64encode(chunk).decode("ascii")
                        self.wfile.write(f"event: audio\ndata: {b64}\n\n".encode())
                        self.wfile.flush()
                except Exception as e:
                    try:
                        self.wfile.write(f"event: error\ndata: {e}\n\n".encode())
                        self.wfile.flush()
                    except OSError:
                        pass
                    return
                finally:
                    conn.close()

            try:
                self.wfile.write(b"event: done\ndata: \n\n")
                self.wfile.flush()
            except OSError:
                pass

        try:
            stream_from_server()
        except BrokenPipeError:
            cancel.set()

    def log_message(self, fmt, *args):
        pass


def main():
    global _tts_url, _pause_ms

    parser = argparse.ArgumentParser(description="Voxtral TTS web demo — server client")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tts-url", default="http://localhost:8000",
                        help="Base URL of the running Voxtral server (default: http://localhost:8000)")
    parser.add_argument("--pause-ms", type=int, default=0,
                        help="Default trailing silence in ms sent per request (default: 0)")
    args = parser.parse_args()

    _tts_url = args.tts_url.rstrip("/")
    _pause_ms = args.pause_ms

    # Verify the server is reachable
    try:
        with urllib.request.urlopen(f"{_tts_url}/v1/voices", timeout=5):
            pass
        print(f"Connected to TTS server at {_tts_url}")
    except Exception as e:
        print(f"Warning: could not reach TTS server at {_tts_url}: {e}")
        print("Continuing anyway — requests will fail until the server is up.")

    print(f"Serving at http://{args.host}:{args.port}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


_tts_url: str = ""
_pause_ms: int = 0

if __name__ == "__main__":
    main()
