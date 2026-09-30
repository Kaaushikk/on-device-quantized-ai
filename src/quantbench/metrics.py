import math


def percentile(values, percent):
    """Linear interpolation between ordered samples."""
    samples = sorted(values)
    if not samples or not 0 <= percent <= 100:
        raise ValueError("Need samples and a percentile between 0 and 100")
    if any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("Durations must be finite and positive")
    position = (len(samples) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return samples[lower] + (samples[upper] - samples[lower]) * (position - lower)


def summarize(durations_ns, batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    p50 = percentile(durations_ns, 50)
    p95 = percentile(durations_ns, 95)
    return {
        "samples": len(durations_ns),
        "batch_size": batch_size,
        "p50_batch_ms": p50 / 1e6,
        "p95_batch_ms": p95 / 1e6,
        "examples_per_second": len(durations_ns) * batch_size * 1e9 / sum(durations_ns),
    }
