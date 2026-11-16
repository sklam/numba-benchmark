"""Timing and measurement of the NumPy and Numba callables from the registry.

Each case is measured in three separate steps:

1. compile: the first Numba call, timed with Numba's ``numba:compile``
   events (see ``compile_time``), so the reported compile time excludes
   executing the freshly compiled kernel;
2. warm-up: one untimed call per arm, so the loop below times compiled code;
3. benchmark loop: ``REPEATS`` interleaved timed batches of ``inner`` calls
   per arm, keeping each arm's fastest average.
"""

import math
import platform
import sys
import time

from cases import build_case, seed_for, select_cases

TARGET_NS = 50_000_000
REPEATS = 5
MAX_INNER = 100_000


def measure_compile(fn, args):
    """Compile ``fn(*args)`` and return ``(compile_ns, compile_count)``."""
    from compile_time import CompileTimeTracker

    tracker = CompileTimeTracker()
    with tracker.monitor():
        fn(*args)
    return int(tracker.compile_time * 1e9), tracker.compile_count


def calibrate(fn, args):
    """Number of calls that should take about ``TARGET_NS`` nanoseconds."""
    started = time.perf_counter_ns()
    fn(*args)
    single_ns = max(time.perf_counter_ns() - started, 1)
    return max(1, min(TARGET_NS // single_ns, MAX_INNER))


def _batch(fn, args, inner):
    """Average ns per ``fn(*args)`` call over one timed batch of ``inner`` calls."""
    started = time.perf_counter_ns()
    for _ in range(inner):
        fn(*args)
    return (time.perf_counter_ns() - started) / inner


def measure_pair(a_fn, b_fn, args):
    """Interleaved A/B benchmark; returns (a_ns, b_ns) best batch averages."""
    a_fn(*args)
    b_fn(*args)  # warm-up calls; the numba kernel is already compiled

    a_inner = calibrate(a_fn, args)
    b_inner = calibrate(b_fn, args)
    a_best = b_best = math.inf

    for _ in range(REPEATS):
        a_best = min(a_best, _batch(a_fn, args, a_inner))
        b_best = min(b_best, _batch(b_fn, args, b_inner))

    return int(a_best), int(b_best)


def collect_sysinfo():
    """The `numba --sys-json` payload, or an error record if it cannot be read."""
    try:
        from numba.misc.numba_sysinfo import get_sysinfo

        info = get_sysinfo()
    except Exception as exc:  # sysinfo must never fail a benchmark run
        return {"error": f"{type(exc).__name__}: {exc}"}
    for key in ("Start", "Start UTC"):
        value = info.get(key)
        if hasattr(value, "isoformat"):
            info[key] = value.isoformat()
    return info


def run_benchmarks():
    try:
        import llvmlite
        import numba
        import numpy as np
    except ImportError as exc:
        sys.exit(f"run requires numpy, numba, and scipy: {exc}")

    runnable, skipped = select_cases(numba.__version__)

    rows = []
    for case in runnable:
        numpy_fn, numba_fn, args = build_case(
            case, np.random.RandomState(seed_for(case.id))
        )

        # The first Numba call compiles the kernel, timed through Numba's
        # events. It must precede measure_pair, whose warm-up call assumes
        # a compiled kernel.
        compile_ns, compile_count = measure_compile(numba_fn, args)
        numpy_ns, numba_ns = measure_pair(numpy_fn, numba_fn, args)

        rows.append(
            {
                "id": case.id,
                "order": case.order,
                "numpy_ns": numpy_ns,
                "numba_ns": numba_ns,
                "compile_ns": compile_ns,
                "compile_count": compile_count,
                "speedup": round(numpy_ns / numba_ns, 3),
            }
        )
        print(
            f"{case.id:32s} numpy {numpy_ns / 1e6:9.3f} ms  "
            f"numba {numba_ns / 1e6:9.3f} ms  "
            f"{numpy_ns / numba_ns:6.2f}x  "
            f"compile {compile_ns / 1e6:9.3f} ms ({compile_count}x)",
            file=sys.stderr,
        )

    for skip in skipped:
        print(f"{skip['id']:32s} skipped: {skip['reason']}", file=sys.stderr)

    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
        "sysinfo": collect_sysinfo(),
    }
    return rows, skipped, environment
