"""Integrate measured power. Synthetic inputs are tests, never measurement evidence."""
import math


def integrate_power(samples):
    rows = list(samples)
    if len(rows) < 2:
        raise ValueError('At least two measured samples are required')
    if any(not math.isfinite(t) or not math.isfinite(w) or w <= 0 for t, w in rows):
        raise ValueError('Timestamps must be finite; measured watts must be finite and positive')
    if any(right[0] <= left[0] for left, right in zip(rows, rows[1:])):
        raise ValueError('Timestamps must strictly increase')
    joules = sum((right[0]-left[0]) * (left[1]+right[1])/2 for left, right in zip(rows, rows[1:]))
    return {'duration_seconds': rows[-1][0]-rows[0][0], 'energy_joules': joules,
            'mean_watts': joules/(rows[-1][0]-rows[0][0]), 'samples': len(rows)}
