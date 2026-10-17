import time
import tracemalloc
import json
import pandas as pd
from pathlib import Path
from src.analysis.triage import score_candidates

def run_system_benchmark():
    tracemalloc.start()
    start_time = time.perf_counter()

    data_path = Path("data/processed/scored_candidates.csv")
    if not data_path.exists():
        print("Scored candidate data not available for benchmark.")
        return

    df = pd.read_csv(data_path)
    
    # Benchmark Inference Throughput
    inference_start = time.perf_counter()
    _ = score_candidates(df.copy())
    inference_duration = time.perf_counter() - inference_start

    total_duration = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    throughput = len(df) / (inference_duration + 1e-9)

    print("\n" + "="*50)
    print("⚡ SYSTEM EFFICIENCY & THROUGHPUT BENCHMARK")
    print("="*50)
    print(f"Evaluated Records:        {len(df)}")
    print(f"Inference Latency:        {inference_duration*1000:.2f} ms")
    print(f"Throughput:               {throughput:.1f} records/sec")
    print(f"Peak Memory Allocated:    {peak_mem / (1024 * 1024):.2f} MB")
    print(f"Total Workflow Latency:   {total_duration*1000:.2f} ms")
    print("="*50)

if __name__ == "__main__":
    run_system_benchmark()
