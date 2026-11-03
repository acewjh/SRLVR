from __future__ import annotations
import argparse, json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
from .aggregate import discover, summarize

def make_handler(root: Path):
    page = Path(__file__).with_name('index.html')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = unquote(self.path.split('?', 1)[0])
            try:
                if path in ('/', '/index.html'): self._send(page.read_bytes(), 'text/html; charset=utf-8')
                elif path == '/api/runs': self._json(discover(root))
                elif path == '/api/overview': self._json({'generated': time.time(), 'budget': {}, 'used': {}, 'used_total': 0, 'running': [], 'plan': {'title':'opsd-slime 训练看板', 'updated':'', 'objective':'训练过程、轨迹质量和训练指标的本地可视化', 'stages':[], 'phases':[], 'decision_gates':[], 'baselines':{}, 'industry_reference':[], 'milestones':[], 'known_issues':[]}})
                elif path.startswith('/api/run/'):
                    rel = path[len('/api/run/'):].strip('/')
                    run = root / rel
                    # NOT run.resolve() for the containment check: a run may be a
                    # SYMLINK into another tree (the colleague's runs/ is mounted
                    # read-only this way for side-by-side comparison), and resolving
                    # first moves it outside root so every such run 404s. Guard on
                    # the relative path instead, which still blocks traversal.
                    if '..' in rel.split('/') or not run.is_dir(): self._json({'error':'unknown run'}, 404)
                    else: self._json(summarize(run.resolve()))
                else: self._json({'error':'not found'}, 404)
            except Exception as exc: self._json({'error': f'{type(exc).__name__}: {exc}'}, 500)
        def _json(self, value, status=200): self._send(json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8', status)
        def _send(self, data, content_type, status=200):
            self.send_response(status); self.send_header('Content-Type', content_type); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        def log_message(self, *args): pass
    return Handler

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='runs'); p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=8765); a=p.parse_args(); root=Path(a.root).expanduser().resolve(); print(f'Open http://{a.host}:{a.port}/'); ThreadingHTTPServer((a.host,a.port),make_handler(root)).serve_forever()
if __name__ == '__main__': main()
