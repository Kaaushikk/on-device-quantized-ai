"""Audit extension trace counts and inputs before generating the separate report."""
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from quantbench.common import read_json, write_json
from quantbench.metrics import summarize

folder = ROOT / 'results/accelerators'
settings = read_json(ROOT / 'configs/accelerator.json')
trace = read_json(ROOT / 'results/accelerator-inputs.json')
measurements = {}
total = 0
for device in ('ORT_CPU', 'GPU', 'NPU'):
    measurements[device] = {}
    for repetition in range(3):
        rows = [json.loads(line) for line in (folder / f'timings-{device}-{repetition}.jsonl').read_text().splitlines()]
        if len(rows) != 600:
            raise ValueError('Expected 600 records per process')
        total += len(rows)
        manifest = read_json(folder / (f'control-{repetition}.json' if device == 'ORT_CPU' else f'manifest-{device}-{repetition}.json'))
        if manifest.get('model_sha256') != settings['graph_sha256'] or manifest['performance_input_sha256'] != trace['performance_sha256']:
            raise ValueError('Graph/input mismatch')
        if device != 'ORT_CPU':
            if manifest['profiling_during_timing']:
                raise ValueError('Profiling enabled during timing')
            for compile_record in manifest['compiles']:
                actual = compile_record['execution_devices']
                if not actual or any(not item.startswith(device) for item in actual):
                    raise ValueError('Timing fallback check failed')
        for length in (32, 128, 256):
            selected = [row for row in rows if row['length'] == length]
            if len(selected) != 200 or [row['iteration'] for row in selected] != list(range(200)):
                raise ValueError('Iteration mismatch')
            pool = trace['performance_pools'][str(length)]
            for i, row in enumerate(selected):
                if row['device'] != device or row['repetition'] != repetition:
                    raise ValueError('Device/repetition mismatch')
                if any(row[key] != value for key, value in pool[i % len(pool)].items()):
                    raise ValueError('Timing fixture mismatch')
            result = summarize([row['duration_ns'] for row in selected], 1)
            measurements[device].setdefault(length, []).append(result)

lines = ['# Optional extension results', '',
         'These results supplement the completed core study. They preserve negative findings.', '',
         '## Static INT8', '',
         'Development accuracy fell from 91.5% to 87.0% for both static candidates. '
         'The selected constant-weight candidate reached 87.95% final accuracy versus '
         '90.92% FP32: a 2.98 percentage-point loss (paired bootstrap 95% interval '
         '[-5.06, -1.04] points). It fails the one-point gate and is not the default. '
         'Calibration used 100 development examples and zero final examples.', '',
         '## Laptop GPU/NPU', '',
         'Intel Arc and Intel AI Boost were selected explicitly in OpenVINO 2025.3.0; '
         'no AUTO/HETERO CPU fallback was used. Each scored 90.92% on all 672 final '
         'examples. GPU/NPU request FP16; the CPU control uses ONNX Runtime 1.22.1 '
         'FP32 with one thread. Comparisons combine runtime, precision and hardware changes.', '',
         '| Length | ORT CPU p50/p95 ms | GPU p50/p95 ms | NPU p50/p95 ms | GPU / NPU p50 speedup |',
         '| --- | --- | --- | --- | --- |']
for length in (32, 128, 256):
    pairs = {device: [statistics.median(row[key] for row in measurements[device][length])
                      for key in ('p50_batch_ms', 'p95_batch_ms')] for device in measurements}
    cpu, gpu, npu = (pairs[device] for device in ('ORT_CPU', 'GPU', 'NPU'))
    lines.append(f'| {length} | {cpu[0]:.2f} / {cpu[1]:.2f} | {gpu[0]:.2f} / {gpu[1]:.2f} | {npu[0]:.2f} / {npu[1]:.2f} | {cpu[0]/gpu[0]:.2f}x / {cpu[0]/npu[0]:.2f}x |')
lines += ['', f'Audited {total:,} timing records: 200 calls × 3 lengths × 3 repetitions × '
          '3 configurations, after 20 warmups per shape. Table entries are medians '
          'of three process-level p50/p95 values; synchronous inference includes '
          'host/device transfer and excludes tokenization. Frozen fixture IDs and '
          'graph/input hashes match across configurations. Long inputs are synthetic '
          'repeated text, as in the core study.', '',
          'GPU/NPU repetition order was rotated. CPU controls ran afterward, so time, '
          'temperature and device-specific threading remain confounders. No confidence '
          'interval or energy-efficiency claim is attached to these speedups. '
          'The OpenVINO CPU process exceeded a 300-second timeout after saving a '
          'passing quality artifact (90.92%, zero changed labels, 0.83-second '
          'compilation). Its exit failure remains unresolved; it supplies no timing '
          'baseline. Compilation time is recorded separately from inference.', '']
for device in ('GPU', 'NPU'):
    quality = read_json(folder / f'quality-{device}.json')
    if quality['row_ids'] != settings['row_ids'] or quality['n'] != 672:
        raise ValueError('Quality rows differ')
    if not quality['manifest']['execution_devices'] or any(not item.startswith(device) for item in quality['manifest']['execution_devices']):
        raise ValueError('Quality fallback check failed')
    lines.append(f'{device}: {quality["changed_predictions_vs_ort_fp32"]} predicted labels changed versus '
                 f'ORT FP32; maximum absolute logit difference '
                 f'{quality["max_absolute_logit_error_vs_ort_fp32"]:.5f}. Equal accuracy does not imply equal logits.')
    lines.append('')
api = read_json(ROOT / 'results/api-check.json')
lines += ['## Local API', '',
          'The real dynamic INT8 service passed readiness, positive/negative labels, '
          'blank-input rejection and wrong-variant rejection. The unit suite separately '
          'checks lifecycle, bounds, metadata and concurrent busy responses.', '',
          f'For two short texts per request, 200 localhost HTTP roundtrips after 20 warmups '
          f'gave p50 {api["timings"]["p50_batch_ms"]:.2f} ms and p95 '
          f'{api["timings"]["p95_batch_ms"]:.2f} ms. This is a single-process service '
          'check with HTTP overhead; it is not comparable to the core fixed-length timings.', '',
          '## Energy and mobile', '',
          'No energy or phone measurement is available. The exposed Windows power '
          'sensor returns unusable zero readings on AC. An external measured-power '
          'CSV importer is implemented and arithmetic/invalid-input tests pass; '
          'synthetic tests are not energy evidence.', '',
          'See `docs/EXTENSIONS.md` for commands and `docs/WORK_LOG.md` for debugging. '
          'Raw traces and manifests are in `results/accelerators`; capability evidence '
          'is in `results/hardware-extensions.json`.', '']
write_json(folder / 'audit.json', {'passed': True, 'records': total, 'configurations': list(measurements),
                                'checks': ['record counts', 'iteration coverage', 'fixture identity', 'graph/input hashes', 'quality row identity', 'explicit quality devices']})
(ROOT / 'reports/EXTENSIONS.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'Extension report: {total} timing records audited')
