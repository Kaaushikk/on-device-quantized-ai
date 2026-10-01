"""Import an external, calibrated meter trace; no estimated laptop power."""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from quantbench.common import sha256, write_json
from quantbench.energy import integrate_power

parser = argparse.ArgumentParser()
parser.add_argument('csv', type=Path)
parser.add_argument('--meter', required=True, help='Meter model and calibration details')
parser.add_argument('--scope', required=True, help='What the meter measures and the workload/window')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if not args.meter.strip() or not args.scope.strip():
    parser.error('Meter and scope must be nonblank')
with args.csv.open(newline='', encoding='utf-8-sig') as stream:
    reader = csv.DictReader(stream)
    if reader.fieldnames != ['time_seconds', 'watts']:
        raise ValueError('CSV headers must be time_seconds,watts')
    samples = [(float(row['time_seconds']), float(row['watts'])) for row in reader]
write_json(args.output, {**integrate_power(samples), 'meter': args.meter, 'scope': args.scope,
                        'csv_sha256': sha256(args.csv), 'method': 'trapezoidal integration; gross energy, no idle subtraction'})
