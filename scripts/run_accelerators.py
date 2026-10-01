"""Run explicit devices sequentially, preserving failures and independent repetitions."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from quantbench.common import write_json


def main():
    executable = ROOT / '.accelerator-venv/Scripts/python.exe'
    folder = ROOT / 'results/accelerators'
    folder.mkdir(exist_ok=True)
    outcomes = []
    passed = []

    def run(device, arguments):
        command = [str(executable), str(ROOT / 'scripts/accelerator_worker.py'), '--device', device, *arguments]
        try:
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
            record = {'device': device, 'arguments': arguments, 'exit_code': process.returncode,
                      'stdout': process.stdout, 'stderr': process.stderr}
        except subprocess.TimeoutExpired as error:
            def decode(value):
                return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value
            record = {'device': device, 'arguments': arguments, 'exit_code': None, 'error': '300-second timeout',
                      'stdout': decode(error.stdout), 'stderr': decode(error.stderr)}
        outcomes.append(record)
        write_json(folder / 'execution-log.json', outcomes)
        print(f'{device} {arguments}: exit={record["exit_code"]}', flush=True)
        return record['exit_code'] == 0

    for device in ('CPU', 'GPU', 'NPU'):
        if run(device, ['--quality-only']):
            passed.append(device)
    for repetition in range(3):
        order = passed[repetition:] + passed[:repetition]
        for device in order:
            run(device, ['--repetition', str(repetition)])


if __name__ == '__main__':
    main()
