"""A minimal browser UI for the agent: paste a customer message, see the
pipeline's output stage by stage.

Stdlib only. The repository deliberately avoids web dependencies (Python 3.14
wheels are thin, and `make reproduce` must not grow an install surface), so
this is `http.server` rather than Flask or FastAPI.

    python tools/ui.py            # http://127.0.0.1:8000
    python tools/ui.py --port 9000 --no-llm-router

The retriever is loaded lazily on the first request because it reads ~10k
threads, and it is cached afterwards by `retrieve.load_retriever`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundscore import config  # noqa: E402
from groundscore.agent import SupportAgent  # noqa: E402
from groundscore.ingest import read_jsonl  # noqa: E402
from groundscore.retrieve import load_retriever  # noqa: E402

_agent_lock = threading.Lock()
_agent: SupportAgent | None = None
_ARGS = argparse.Namespace(use_llm_router=True, retrieval=True, backend=None)


def get_agent() -> SupportAgent:
    """Built once, on first use. Loading the index takes a few seconds."""
    global _agent
    with _agent_lock:
        if _agent is None:
            retriever = load_retriever(_ARGS.backend) if _ARGS.retrieval else None
            _agent = SupportAgent(
                retriever,
                use_llm_router=_ARGS.use_llm_router,
                name="ui",
            )
        return _agent


def sample_messages(n: int = 6) -> list[str]:
    """A handful of golden-pool messages, so the UI is usable without typing.

    Golden-pool threads are excluded from the retrieval index, so these are not
    messages the agent can look up verbatim.
    """
    try:
        pool = [
            t["customer_msg"] for t in read_jsonl(config.THREADS_PATH)
            if t.get("split") == config.SPLIT_GOLDEN_POOL
            and 40 < len(t["customer_msg"]) < 240
        ]
    except FileNotFoundError:
        return []
    return random.Random(0).sample(pool, min(n, len(pool)))


PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ground-score</title>
<style>
  :root {
    --bg: #fbfbf9; --panel: #fff; --ink: #1b1b19; --muted: #6b6b66;
    --line: #e3e3de; --accent: #3d5afe; --auto: #1a7f4b; --esc: #b4471f;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #17171a; --panel: #1f1f23; --ink: #ececea; --muted: #9a9a94;
      --line: #32323a; --accent: #8fa2ff; --auto: #5fd39b; --esc: #ff9b6a;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 32px 20px 64px; background: var(--bg); color: var(--ink);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 13px; margin: 0 0 24px; }
  textarea {
    width: 100%; min-height: 110px; padding: 12px 14px; resize: vertical;
    background: var(--panel); color: var(--ink); border: 1px solid var(--line);
    border-radius: 10px; font: inherit;
  }
  textarea:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .row { display: flex; gap: 10px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  button {
    background: var(--accent); color: #fff; border: 0; border-radius: 8px;
    padding: 9px 18px; font: inherit; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: .55; cursor: default; }
  .hint { color: var(--muted); font-size: 12px; }
  .samples { margin: 14px 0 0; display: flex; flex-direction: column; gap: 6px; }
  .samples button {
    background: transparent; color: var(--muted); border: 1px solid var(--line);
    text-align: left; font-weight: 400; font-size: 13px; padding: 7px 10px;
    white-space: normal;
  }
  .samples button:hover { color: var(--ink); border-color: var(--accent); }
  section {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 16px 18px; margin-top: 16px;
  }
  h2 {
    font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
    color: var(--muted); margin: 0 0 10px; font-weight: 600;
  }
  .badges { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .badge {
    border: 1px solid var(--line); border-radius: 999px; padding: 3px 11px;
    font-size: 12px; font-weight: 600;
  }
  .badge.auto { color: var(--auto); border-color: var(--auto); }
  .badge.escalate { color: var(--esc); border-color: var(--esc); }
  .reply { white-space: pre-wrap; }
  dl {
    display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px;
    margin: 0; font-size: 13px;
  }
  dt { color: var(--muted); }
  dd { margin: 0; }
  .ex { border-top: 1px solid var(--line); padding: 10px 0 0; margin-top: 10px; font-size: 13px; }
  .ex:first-of-type { border-top: 0; padding-top: 0; margin-top: 0; }
  .ex .meta { color: var(--muted); font-size: 12px; margin-bottom: 3px; }
  .ex .msg { margin: 0 0 4px; }
  .ex .rep { margin: 0; color: var(--muted); }
  .err {
    color: var(--esc); white-space: pre-wrap; font-size: 12px;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  }
  [hidden] { display: none !important; }
</style>
<main>
  <h1>ground-score</h1>
  <p class="sub">retrieve &rarr; classify &rarr; draft &rarr; route. One message at a time.</p>

  <textarea id="msg" placeholder="Paste a customer message to @AmazonHelp..."></textarea>
  <div class="row">
    <button id="go">Run</button>
    <span class="hint" id="status"></span>
  </div>
  <div class="samples" id="samples"></div>

  <section id="error" hidden><h2>Error</h2><div class="err" id="errbody"></div></section>

  <div id="out" hidden>
    <section>
      <h2>Routing</h2>
      <div class="badges">
        <span class="badge" id="action"></span>
        <span class="badge" id="intent"></span>
        <span class="badge" id="conf"></span>
      </div>
      <dl style="margin-top:12px">
        <dt>reason</dt><dd id="reason"></dd>
        <dt>rule</dt><dd id="rule"></dd>
        <dt>decided by</dt><dd id="by"></dd>
        <dt>top similarity</dt><dd id="sim"></dd>
      </dl>
    </section>

    <section>
      <h2>Draft reply</h2>
      <div class="reply" id="reply"></div>
    </section>

    <section>
      <h2>Classification rationale</h2>
      <div id="rationale"></div>
    </section>

    <section>
      <h2>Retrieved precedent</h2>
      <div id="exemplars"></div>
    </section>
  </div>
</main>
<script>
  const $ = (id) => document.getElementById(id);
  const text = (id, v) => { $(id).textContent = v; };

  fetch("/api/samples").then((r) => r.json()).then((list) => {
    for (const s of list) {
      const b = document.createElement("button");
      b.textContent = s;
      b.onclick = () => { $("msg").value = s; run(); };
      $("samples").appendChild(b);
    }
  });

  async function run() {
    const message = $("msg").value.trim();
    if (!message) return;
    $("go").disabled = true;
    text("status", "running (the first call loads the index)...");
    $("error").hidden = true;
    try {
      const res = await fetch("/api/handle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      render(data);
      text("status", "");
    } catch (e) {
      $("out").hidden = true;
      $("error").hidden = false;
      text("errbody", String(e.message || e));
      text("status", "");
    } finally {
      $("go").disabled = false;
    }
  }

  function render(d) {
    $("out").hidden = false;
    const act = $("action");
    act.textContent = d.action;
    act.className = "badge " + d.action;
    text("intent", d.intent + (d.secondary_intent ? " / " + d.secondary_intent : ""));
    text("conf", "confidence " + d.confidence.toFixed(2));
    text("reason", d.reason || "-");
    text("rule", d.triggered_rule || "-");
    text("by", d.decided_by);
    text("sim", d.max_similarity.toFixed(4));
    text("reply", d.reply || "(no reply drafted)");
    text("rationale", d.rationale || "-");

    const box = $("exemplars");
    box.textContent = "";
    if (!d.exemplars.length) {
      box.textContent = "No exemplars (retrieval disabled, or the index is empty).";
      return;
    }
    for (const e of d.exemplars) {
      const div = document.createElement("div");
      div.className = "ex";
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = "sim " + e.similarity.toFixed(3) + " · " + e.thread_id +
        (e.is_deflection ? " · deflection" : "");
      const msg = document.createElement("p");
      msg.className = "msg";
      msg.textContent = e.customer_msg;
      const rep = document.createElement("p");
      rep.className = "rep";
      rep.textContent = "→ " + e.brand_reply;
      div.append(meta, msg, rep);
      box.appendChild(div);
    }
  }

  $("go").onclick = run;
  $("msg").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
  });
</script>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "groundscore-ui"

    def log_message(self, fmt, *args):
        if self.path.startswith("/api/"):
            sys.stderr.write("%s %s\n" % (self.command, self.path))

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/samples":
            self._json(200, sample_messages())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/handle":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            message = (payload.get("message") or "").strip()
            if not message:
                self._json(400, {"error": "empty message"})
                return
            out = get_agent().handle("ui", message)
            self._json(200, out.as_dict())
        except Exception as exc:
            traceback.print_exc()
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser UI for the support agent.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-llm-router", dest="use_llm_router", action="store_false",
                        help="rules only; skips the router's LLM call")
    parser.add_argument("--no-retrieval", dest="retrieval", action="store_false",
                        help="run the ungrounded ablation pipeline")
    parser.add_argument("--backend", default=None,
                        help="embedding backend for the index; 'tfidf' needs no Ollama "
                             "(default: configs/brand.yaml)")
    parser.add_argument("--no-open", dest="open_browser", action="store_false")
    global _ARGS
    _ARGS = parser.parse_args()

    url = f"http://{_ARGS.host}:{_ARGS.port}"
    print(f"ground-score UI on {url}")
    print("Drafting needs Ollama running, or GROUNDSCORE_PROVIDER=gemini with a key.")
    if _ARGS.open_browser:
        threading.Timer(0.5, webbrowser.open, args=[url]).start()
    ThreadingHTTPServer((_ARGS.host, _ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
