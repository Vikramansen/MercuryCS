import time
import requests
import numpy as np

API_URL = "http://localhost:8000/chat"

def measure_latency(queries: list[str], iterations: int = 5):
    latencies = []
    
    print(f"Benchmarking {len(queries)} queries over {iterations} iterations...")
    
    for query in queries:
        for _ in range(iterations):
            start = time.time()
            try:
                response = requests.post(API_URL, json={"query": query})
                response.raise_for_status()
                # We can also use the server-reported latency
                # server_latency = response.json()['latency_ms']
            except Exception as e:
                print(f"Request failed: {e}")
                continue
            end = time.time()
            latencies.append((end - start) * 1000)
            
    if not latencies:
        print("No successful requests.")
        return

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg = np.mean(latencies)
    
    print(f"\nLatency Stats (ms):")
    print(f"Average: {avg:.2f}")
    print(f"P50: {p50:.2f}")
    print(f"P95: {p95:.2f}")
    print(f"P99: {p99:.2f}")

if __name__ == "__main__":
    test_queries = [
        "How long is shipping?",
        "Can I return my order?",
        "Do you have headphones?",
        "What is the meaning of life?" # Unsupported
    ]
    
    # Note: Ensure the API is running before executing this script
    try:
        measure_latency(test_queries)
    except requests.exceptions.ConnectionError:
        print("Error: API is not running. Please start the API with 'uvicorn api.main:app' first.")
