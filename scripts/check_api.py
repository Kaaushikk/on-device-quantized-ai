"""Live localhost service check with the real, hash-verified dynamic INT8 model."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from quantbench.common import write_json
from quantbench.metrics import summarize


def main():
    with socket.socket() as socket_handle:
        socket_handle.bind(('127.0.0.1', 0))
        port = socket_handle.getsockname()[1]
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    log = ROOT / 'artifacts/api-check.log'
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, '-m', 'quantbench.cli', 'serve', '--port', str(port)],
                                   cwd=ROOT, env=environment, stdout=stream, stderr=stream,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=30, trust_env=False) as client:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f'Service exited; see {log}')
                    try:
                        response = client.get('/health/ready')
                        if response.status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.25)
                else:
                    raise RuntimeError('Service readiness timed out')
                payload = {'variant': 'onnx_int8', 'texts': ['I love this wonderful movie.', 'I hate this terrible movie.']}
                response = client.post('/v1/predict', json=payload)
                response.raise_for_status()
                body = response.json()
                assert [row['label'] for row in body['predictions']] == ['POSITIVE', 'NEGATIVE']
                assert len(body['artifact_hash']) == 64
                assert client.post('/v1/predict', json={'variant': 'onnx_int8', 'texts': [' ']}).status_code == 422
                assert client.post('/v1/predict', json={'variant': 'onnx_fp32', 'texts': ['hello']}).status_code == 400
                for _ in range(20):
                    client.post('/v1/predict', json=payload).raise_for_status()
                durations = []
                for _ in range(200):
                    start = time.perf_counter_ns()
                    client.post('/v1/predict', json=payload).raise_for_status()
                    durations.append(time.perf_counter_ns() - start)
                write_json(ROOT / 'results/api-check.json', {'passed': True, 'response': body,
                           'scope': 'localhost HTTP roundtrip; two short texts, single process, 20 warmups, 200 calls; not comparable to core inference timings',
                           'timings': summarize(durations, 2), 'duration_ns': durations})
                print('Live API: labels, readiness, validation and 200 HTTP requests passed', flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == '__main__':
    main()
