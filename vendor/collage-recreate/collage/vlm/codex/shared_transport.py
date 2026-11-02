"""Batch-owned, multiplexed Node bridge; never installed as a global transport."""
from concurrent.futures import Future, TimeoutError as FutureTimeout
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading

from .transport import CodexError, ERRORS, MAX_RESPONSE, check_runtime, worker_request

BRIDGE = Path(__file__).with_name('shared_server.mjs')


class SharedTransport:
    def __init__(self, timeout=600):
        self.timeout = timeout
        self.guard = threading.Lock()
        self.writer = threading.Lock()
        self.pending = {}
        self.sequence = 0
        self.process = None
        self.failure = None
        self.closed = False
        self.temp = None
        self.reader = None

    def __enter__(self):
        return self

    def _start(self):
        # Called under guard; lazy startup allows cached-only runs without Node/auth.
        if self.closed or self.failure:
            raise CodexError(self.failure or ERRORS['runtime'])
        if self.process is not None:
            return
        node = check_runtime()
        self.temp = tempfile.TemporaryDirectory(prefix='collage-batch-server-')
        flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
        try:
            self.process = subprocess.Popen([node, str(BRIDGE)], cwd=self.temp.name,
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, **flags)
        except OSError:
            self.temp.cleanup()
            self.temp = None
            raise CodexError(ERRORS['runtime']) from None
        self.reader = threading.Thread(target=self._read, name='analysis-batch-results', daemon=True)
        self.reader.start()

    def _fail(self, message):
        with self.guard:
            self.failure = message
            pending, self.pending = self.pending, {}
        for future in pending.values():
            future.set_exception(CodexError(message))

    def _read(self):
        try:
            while True:
                line = self.process.stdout.readline(MAX_RESPONSE + 1)
                if not line:
                    break
                if len(line) > MAX_RESPONSE:
                    raise ValueError('Oversized response')
                document = json.loads(line)
                with self.guard:
                    future = self.pending.pop(document['id'], None)
                if future is not None:
                    future.set_result(document['result'])
        except (ValueError, KeyError, TypeError, OSError):
            self._fail(ERRORS['invalid'])
        finally:
            self._fail(ERRORS['failed'])

    def __call__(self, payload):
        with tempfile.TemporaryDirectory(prefix='collage-batch-case-') as folder:
            request = worker_request(payload, Path(folder), self.timeout)
            future = Future()
            with self.guard:
                self._start()
                self.sequence += 1
                identifier = self.sequence
                self.pending[identifier] = future
            try:
                data = json.dumps({'id': identifier, 'request': request}, ensure_ascii=False).encode('utf-8') + b'\n'
                with self.writer:
                    self.process.stdin.write(data)
                    self.process.stdin.flush()
                # Node enforces the inference deadline and interrupts only that turn.
                # Allow bounded RPC cleanup before treating the bridge as unresponsive.
                result = future.result(timeout=self.timeout + 65)
            except (BrokenPipeError, OSError):
                self._fail(ERRORS['failed'])
                raise CodexError(ERRORS['failed']) from None
            except FutureTimeout:
                self._fail(ERRORS['failed'])
                raise CodexError(ERRORS['failed']) from None
            finally:
                with self.guard:
                    self.pending.pop(identifier, None)
            if not isinstance(result, dict):
                raise CodexError(ERRORS['invalid'])
            if result.get('error'):
                code = result['error']
                raise CodexError(ERRORS.get(code if isinstance(code, str) else '', ERRORS['failed']))
            if not result.get('choices'):
                raise CodexError(ERRORS['invalid'])
            return result

    def close(self):
        with self.guard:
            if self.closed:
                return
            self.closed = True
        self._fail(ERRORS['failed'])
        if self.process is not None:
            try:
                with self.writer:
                    self.process.stdin.write(b'{"method":"shutdown"}\n')
                    self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                if self.process.poll() is None and os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                                   creationflags=subprocess.CREATE_NO_WINDOW, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=10, check=False)
                elif self.process.poll() is None:
                    import signal
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait(timeout=10)
            finally:
                self.reader.join(timeout=5)
                for stream in (self.process.stdin, self.process.stdout):
                    try:
                        stream.close()
                    except OSError:
                        pass  # A dead Windows pipe may fail while flushing on close.
                self.temp.cleanup()

    def __exit__(self, *args):
        self.close()
