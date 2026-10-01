"""Contemporaneous ONNX Runtime FP32 control on the accelerator study's exact inputs."""
import json
import sys
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from quantbench.common import ARTIFACTS, read_json, sha256, write_json
from quantbench.metrics import summarize

settings = read_json(ROOT / 'configs/accelerator.json')
trace = read_json(ROOT / 'results/accelerator-inputs.json')
assert sha256(ARTIFACTS / 'model.fp32.onnx') == settings['graph_sha256']
assert sha256(ARTIFACTS / 'accelerator-performance.npz') == trace['performance_sha256']
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session = ort.InferenceSession(str(ARTIFACTS / 'model.fp32.onnx'), options, providers=['CPUExecutionProvider'])
pool = np.load(ARTIFACTS / 'accelerator-performance.npz')
records, summaries = [], []
repetition = int(sys.argv[1])
for length in settings['performance_lengths']:
    count = len(pool[f'input_ids_{length}'])
    inputs = [{key: pool[f'{key}_{length}'][i:i+1] for key in ('input_ids', 'attention_mask')} for i in range(count)]
    for i in range(settings['warmup']):
        session.run(None, inputs[i % count])
    durations = []
    for i in range(settings['iterations']):
        start = perf_counter_ns()
        output = session.run(None, inputs[i % count])[0]
        duration = perf_counter_ns()-start
        assert np.isfinite(output).all()
        durations.append(duration)
        records.append({'device': 'ORT_CPU', 'repetition': repetition, 'length': length, 'iteration': i,
                        'duration_ns': duration, **trace['performance_pools'][str(length)][i % count]})
    summaries.append({'length': length, **summarize(durations, 1)})
folder = ROOT / 'results/accelerators'
write_json(folder / f'control-{repetition}.json', {'engine': 'ONNX Runtime', 'version': ort.__version__,
           'precision': 'FP32', 'threads': 1, 'providers': session.get_providers(), 'summaries': summaries,
           'model_sha256': settings['graph_sha256'], 'performance_input_sha256': trace['performance_sha256']})
(folder / f'timings-ORT_CPU-{repetition}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records), encoding='utf-8')
print(f'ORT CPU control {repetition}: {len(records)} records', flush=True)
