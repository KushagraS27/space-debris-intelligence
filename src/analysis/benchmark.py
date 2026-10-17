import time
import tracemalloc
import pandas as pd
from pathlib import Path
from src.analysis.triage import score_candidates

def run_system_benchmark():
    data_path = Path("data/processed/scored_candidates.csv")
    if not data_path.exists():
        print("No data available for benchmarking.")
        return

    df = pd.read_csv(data_path)
    
    print("\n" + "="*60)
    print("⚡ STEP 4: SYSTEM EFFICIENCY & THROUGHPUT BENCHMARK")
    print("="*60)
    
    # Start tracking memory and time
    tracemalloc.start()
    start_time = time.perf_counter()

    # Run the ML ensemble inference
    scored_df = score_candidates(df.copy())
    
    # Stop tracking
    inference_duration = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    throughput = len(df) / (inference_duration + 1e-9)

    print(f"Evaluated Records:        {len(df)} encounter pairs")
    print(f"Total Inference Latency:  {inference_duration*1000:.2f} ms")
    print(f"Throughput:               {throughput:.1f} records/sec")
    print(f"Peak Memory Allocated:    {peak_mem / (1024 * 1024):.2f} MB")
    print("="*60)

if __name__ == "__main__":
    run_system_benchmark()
