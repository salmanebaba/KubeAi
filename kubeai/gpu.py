"""Optional single-device nvidia-smi exporter, explicitly mapped to a replica."""

import csv
import io
import math
import subprocess
import threading
import time

from .transport import Handler, Server


def sample(gpu_id):
    command = ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=2)
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) != 1:
        raise ValueError("exporter must select exactly one physical GPU")
    total, free, utilization = map(float, rows[0])
    if not all(math.isfinite(x) for x in (total, free, utilization)) or not 0 <= free <= total or total <= 0 or not 0 <= utilization <= 100:
        raise ValueError("invalid NVIDIA telemetry")
    return {"kubeai_gpu_memory_total_mib": total, "kubeai_gpu_memory_free_mib": free, "kubeai_gpu_utilization_percent": utilization}


def make_server(host="127.0.0.1", port=9401, gpu_id="0"):
    class GPUHandler(Handler):
        def do_GET(self):
            if self.path not in ("/metrics", "/healthz"):
                return self.error(404, "unknown endpoint")
            try:
                values = sample(gpu_id)
            except (OSError, ValueError, subprocess.SubprocessError):
                return self.error(503, "single-device NVIDIA telemetry unavailable")
            if self.path == "/healthz":
                return self.reply(200, {"status": "ok"})
            self.reply(200, "".join(f"# TYPE {name} gauge\n{name} {value}\n" for name, value in values.items()), "text/plain; version=0.0.4")

    return Server((host, port), GPUHandler)
