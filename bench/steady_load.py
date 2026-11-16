import csv
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "http://127.0.0.1:8000/v1/chat/completions"
WORKERS = 4
DURATION = 30
BODY = Path("request.json").read_bytes()

def worker(worker_id, deadline):
    rows = []
    while time.monotonic() < deadline:
        started_at = time.time()
        start = time.perf_counter()
        status, finish, tokens = 0, "error", 0

        try:
            request = urllib.request.Request(
                URL, data=BODY,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                status = response.status
                result = json.load(response)
            finish = result["choices"][0]["finish_reason"]
            tokens = result["usage"]["completion_tokens"]
        except Exception as exc:
            finish = type(exc).__name__

        rows.append((
            started_at, worker_id, status,
            time.perf_counter() - start, finish, tokens,
        ))
    return rows

begin = time.monotonic()
deadline = begin + DURATION

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    rows = [row for batch in pool.map(
        lambda worker_id: worker(worker_id, deadline), range(WORKERS)
    ) for row in batch]

elapsed = time.monotonic() - begin
output = Path("results/steady-c4.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("start_unix", "worker", "http_code",
                     "latency_seconds", "finish_reason", "output_tokens"))
    writer.writerows(rows)

latencies = sorted(row[3] for row in rows if row[2] == 200)
complete = sum(row[2] == 200 and row[4] == "stop" for row in rows)

print(f"Elapsed: {elapsed:.2f} s")
print(f"Requests: {len(rows)}; HTTP 200: {len(latencies)}; completed: {complete}")
print(f"Completed throughput: {complete / elapsed:.2f} requests/s")
if latencies:
    print(f"Median latency: {latencies[len(latencies)//2]:.3f} s")
    print(f"P95 latency: {latencies[math.ceil(.95*len(latencies))-1]:.3f} s")
print(f"Saved: {output}")
