import json
import pickle
import config
import os
import functools
import sqlite3
from collections import defaultdict
from itertools import product
import math
from typing import Dict, List, Tuple, Any, Set

"""
nsys_parser.py (2025‑04‑23)
====================================
* **Public API**
  * `get_avg_timings_by_module(...)`  → returns a **four‑tuple**
  * `get_all_results(...)`            → convenience wrapper used by *tuner.py*
  * `kernel_ordering_to_index(...)`  → encode kernel orderings
  * `index_to_kernel_ordering(...)`  → decode kernel orderings

Returned structure
------------------
```python
(
    avg_standalone,   # {module: µs}
    avg_decdec,       # combined GEMV+DEC window {module: tb ▸ k ▸ µs}
    avg_gemv_only,    # GEMV‑only latency         {module: tb ▸ k ▸ µs}
    avg_dec_only,     # DEC‑only latency          {module: tb ▸ k ▸ µs}
    union_orderings,  # union of all orderings   {module: set of ordering indices}
)
```
All times are **micro‑seconds** (µs).

The ordering indices are encoded as integers, which can be decoded using the `index_to_kernel_ordering(...)` function.
"""

################################################################################
#  SQLite helpers                                                              #
################################################################################

def _open_db(fname: str, *, memcopy: bool = True) -> sqlite3.Connection:
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    disk = sqlite3.connect(fname)
    if not memcopy:
        return disk
    mem = sqlite3.connect(":memory:")
    disk.backup(mem)
    disk.close()
    return mem


def _fetch_kernel_events(dbfile: str, names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    conn = _open_db(dbfile)
    cur = conn.cursor()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name in names:
        cur.execute(
            """
            SELECT start, end, gridX, gridY, gridZ, blockX, blockY, blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE shortName = (SELECT id FROM StringIds WHERE value = ?)
            ORDER BY start
            """,
            (name,),
        )
        out[name] = [
            {
                "start_ns": r[0],
                "end_ns": r[1],
                "grid": (r[2], r[3], r[4]),
                "block": (r[5], r[6], r[7]),
            }
            for r in cur.fetchall()
        ]
    conn.close()
    return out

#################################################################################
#  Kernel ordering encode/decode helpers                                         
#################################################################################


# How to encode the kernel events as integers
_kernel_event_str_to_int = {
    'dec_start': 0,
    'dec_end': 1,
    'gemv_start': 2,
    'gemv_end': 3,
}

# Construct the reverse mapping
_kernel_event_int_to_str = {}
for k, v in _kernel_event_str_to_int.items():
    _kernel_event_int_to_str[v] = k


def _permutation_to_index(perm):
    pool = list(sorted(perm))  # make sure it's in canonical order
    index = 0
    for i, val in enumerate(perm):
        pos = pool.index(val)
        index += pos * math.factorial(len(pool) - 1)
        pool.pop(pos)
    return index


def _index_to_permutation(index, size):
    pool = list(range(size))
    perm = []
    for i in range(size):
        fact = math.factorial(size - 1 - i)
        pos = index // fact
        perm.append(pool[pos])
        pool.pop(pos)
        index %= fact
    return perm


def kernel_ordering_to_index(ordering_str_tuple):
    """Convert an ordering of kernel events to an encoded index."""
    permutation = [_kernel_event_str_to_int[s] for s in ordering_str_tuple]
    return _permutation_to_index(permutation)


def index_to_kernel_ordering(index):
    """Convert an encoded kernel ordering index to the actual kernel event ordering as a tuple of strings."""
    ordering = _index_to_permutation(index, 4)
    return tuple(_kernel_event_int_to_str[i] for i in ordering)


################################################################################
#  Core parser                                                                 #
################################################################################

def _process_profile(
    prefix: str,
    model: str,
    fp: str,
    algo: str,
    bits: int,
) -> Tuple[
    Dict[str, List[int]],
    Dict[str, Dict[int, Dict[int, List[int]]]],
    Dict[str, Dict[int, Dict[int, List[int]]]],
    Dict[str, Dict[int, Dict[int, List[int]]]],
]:
    path = config.get_output_path(prefix, model, fp, algo, bits)
    gemv_name = config.get_gemv_kernel_name(algo, bits)

    events = _fetch_kernel_events(path + ".sqlite", [gemv_name, config.dec_kernel_name])
    gemv_events = events[gemv_name]
    dec_events = events[config.dec_kernel_name]

    with open(path + ".trace") as f:
        traces = [ln.split() for ln in f]
    gemv_traces = [tr for tr in traces if gemv_name in tr]
    dec_traces = [tr for tr in traces if config.dec_kernel_name in tr]

    assert len(gemv_traces) == len(gemv_events), f"GEMV events: {len(gemv_events)}, traces: {len(gemv_traces)}"
    assert len(dec_traces) == len(dec_events), f"DEC events: {len(dec_events)}, traces: {len(dec_traces)}"

    trace_event_pairs = list(zip(gemv_traces, gemv_events)) + list(zip(dec_traces, dec_events))

    modules = config.model_configurations[model]
    standalone: Dict[str, List[int]] = {m: [] for m in modules}
    decdec: Dict[str, Dict[int, Dict[int, List[int]]]] = {m: {} for m in modules}
    gemv_only: Dict[str, Dict[int, Dict[int, List[int]]]] = {m: {} for m in modules}
    dec_only: Dict[str, Dict[int, Dict[int, List[int]]]] = {m: {} for m in modules}
    kernel_event_ordering_index: Dict[str, Dict[int, Dict[int, List[int]]]] = {m: {} for m in modules}

    concurrent: Dict[int, List[Any]] = defaultdict(list)
    debug_warning = False

    for trace, event in trace_event_pairs:
        tag = trace[0]

        if tag == "standalone":
            module_name = trace[2]
            assert module_name in modules, f"Unknown module name in trace: {module_name}"
            duration = int(event["end_ns"]) - int(event["start_ns"])
            standalone[module_name].append(duration)

        elif tag == "concurrent":
            idx = int(trace[1])
            concurrent[idx].append((trace, event))

        elif tag == "debug":
            debug_warning = True
            continue

        elif tag == "warmup":
            continue

        elif tag == "compile":
            continue

        else:
            raise ValueError(f"Unexpected trace tag: {tag}")

    if debug_warning:
        print("WARNING: Debug trace found in profile results. "
              "Running the profiler in debug mode may affect performance.")

    concurrent_pairs = [concurrent[i] for i in range(len(concurrent))]
    assert all(len(p) == 2 for p in concurrent_pairs), "Each concurrent invocation should have exactly 2 entries (GEMV + DEC)."

    for (trace_a, event_a), (trace_b, event_b) in concurrent_pairs:
        if config.dec_kernel_name in trace_a:
            dec_trace, dec_event = trace_a, event_a
            gemv_trace, gemv_event = trace_b, event_b
        else:
            gemv_trace, gemv_event = trace_a, event_a
            dec_trace, dec_event = trace_b, event_b

        module_name = gemv_trace[3]
        num_tb = functools.reduce(lambda x, y: x * y, dec_event["grid"])
        k_chunk = int(dec_trace[3])

        def insert(dct, val):
            dct.setdefault(num_tb, {}).setdefault(k_chunk, []).append(val)

        # Extract start and end times
        gemv_start = int(gemv_event["start_ns"])
        gemv_end = int(gemv_event["end_ns"])
        dec_start = int(dec_event["start_ns"])
        dec_end = int(dec_event["end_ns"])

        # Combined GEMV + DEC
        start_ns = min(dec_start, gemv_start)
        end_ns = max(dec_end, gemv_end)
        insert(decdec[module_name], end_ns - start_ns)

        # GEMV-only
        gemv_duration = gemv_end - gemv_start
        insert(gemv_only[module_name], gemv_duration)

        # DEC-only
        dec_duration = dec_end - dec_start
        insert(dec_only[module_name], dec_duration)

        # Kernel event ordering index
        ordering_helper = [
            (gemv_start, _kernel_event_str_to_int["gemv_start"]),
            (gemv_end, _kernel_event_str_to_int["gemv_end"]),
            (dec_start, _kernel_event_str_to_int["dec_start"]),
            (dec_end, _kernel_event_str_to_int["dec_end"]),
        ]
        ordering_helper.sort(key=lambda x: x[0])
        ordering = [x[1] for x in ordering_helper]
        ordering_index = _permutation_to_index(ordering)
        insert(kernel_event_ordering_index[module_name], ordering_index)

    return standalone, decdec, gemv_only, dec_only, kernel_event_ordering_index

################################################################################
#  Public helpers                                                              #
################################################################################

def _avg(lst: List[int]) -> float:
    return sum(lst) / len(lst) / 1000.0 if lst else float("nan")


def _avg_nested(d: Dict[int, Dict[int, List[int]]]) -> Dict[int, Dict[int, float]]:
    return {num_tb: {k_chunk: _avg(times) for k_chunk, times in inner.items()} for num_tb, inner in d.items()}


def _union_nested(d: Dict[int, Dict[int, List[int]]]) -> Dict[int, Dict[int, Set[int]]]:
    """Preserve full nesting structure, but convert List[int] → Set[int] at the leaf level."""
    return {
        num_tb: {
            k_chunk: set(orderings) for k_chunk, orderings in inner.items()
        }
        for num_tb, inner in d.items()
    }


def get_avg_timings_by_module(
    prefix: str,
    model_name: str,
    fp_dtype: str,
    algo: str,
    bitwidth: int,
    *,
    cache_dir: str = "./cache",
):
    """Cached five‑tuple:
       (standalone avg, decdec avg, gemv avg, dec avg, kernel ordering sets per TB/k_chunk)."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{prefix}_{model_name}_{fp_dtype}_{algo}_{bitwidth}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            raw = pickle.load(f)
    else:
        raw = _process_profile(prefix, model_name, fp_dtype, algo, bitwidth)
        with open(cache_path, "wb") as f:
            pickle.dump(raw, f)

    (
        standalone_raw,
        decdec_raw,
        gemv_raw,
        dec_raw,
        ordering_raw,
    ) = raw

    return (
        {module: _avg(times) for module, times in standalone_raw.items()},
        {module: _avg_nested(nested) for module, nested in decdec_raw.items()},
        {module: _avg_nested(nested) for module, nested in gemv_raw.items()},
        {module: _avg_nested(nested) for module, nested in dec_raw.items()},
        {module: _union_nested(nested) for module, nested in ordering_raw.items()},
    )


def get_all_results(prefix: str, fp_dtype: str):
    """Returns dict[model][algo][bitwidth][module] =
       (standalone avg, decdec avg, gemv avg, dec avg, ordering sets)."""
    results: Dict[str, Any] = {}

    for algo, bitwidth, model_name in product(config.algos, config.bitwidths, config.model_configurations.keys()):
        try:
            print(f"Processing {model_name} {algo} {bitwidth}-bit...")

            (
                avg_standalone_by_module,
                avg_decdec_by_module,
                avg_gemv_by_module,
                avg_dec_by_module,
                ordering_set_by_module,
            ) = get_avg_timings_by_module(prefix, model_name, fp_dtype, algo, bitwidth)

            results.setdefault(model_name, {}).setdefault(algo, {}).setdefault(bitwidth, {})

            for module_name in config.model_configurations[model_name]:
                results[model_name][algo][bitwidth][module_name] = (
                    avg_standalone_by_module[module_name],
                    avg_decdec_by_module[module_name],
                    avg_gemv_by_module[module_name],
                    avg_dec_by_module[module_name],
                    ordering_set_by_module[module_name],
                )

        except FileNotFoundError:
            print("File not found, skipping...")
            continue

    return results