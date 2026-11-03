"""Watch completed JSONL records independently of subprocess stdout buffering."""
from pathlib import Path
import threading

from preview_report import build_report


class LivePreview:
    def __init__(self, output, total, baseline=None, interval=1.0):
        self.output = Path(output)
        self.total = total
        self.baseline = baseline
        self.interval = interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.count = 0
        self.cache = {}
        self.last_size = -1
        self.last_error = None

    def update(self, final=False, stopped=False):
        path = self.output / 'predictions.jsonl'
        size = path.stat().st_size if path.exists() else 0
        if not final and not stopped and size == self.last_size:
            return
        count = build_report(self.output, self.baseline, partial=not final,
                             expected=self.total, cache=self.cache, quiet=True, running=not stopped)
        self.last_size = size
        if count != self.count:
            message = f'[PREVIEW] 已完成 {count}/{self.total} 张，结果已保存：{self.output / "preview.html"}'
            print(message, flush=True)
            with (self.output / 'preview-progress.log').open('a', encoding='utf-8') as f:
                f.write(message + '\n')
            self.count = count

    def _watch(self):
        while not self.stop.is_set():
            try:
                self.update()
            except Exception as exc:
                # Inference/raw output must survive a visualization failure.
                message = f'[PREVIEW WARNING] {type(exc).__name__}: {exc}'
                if message != self.last_error:
                    print(message, flush=True)
                    self.last_error = message
            self.stop.wait(self.interval)

    def start(self):
        self.thread.start()

    def finish(self, success):
        self.stop.set()
        self.thread.join()
        try:
            self.update(final=success, stopped=True)
        finally:
            self.cache.clear()
