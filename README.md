# KubeAI

Adaptive inference routing and GPU resource orchestration: a research prototype for serving large language models under resource constraints.

**Research question:** can telemetry-informed routing and replica management improve delivered performance and energy efficiency while respecting latency, reliability and GPU-capacity constraints?

This project develops and evaluates orchestration policies. vLLM is the model-serving backend; monitoring and the dashboard support experiments.

- **Start here:** setup and usage are documented below.
- **Progress and evidence:** [docs/PROGRESS.md](docs/PROGRESS.md).
- **Demonstration:** [docs/DEMO.md](docs/DEMO.md).
- **Kubernetes:** [deployment guide](deploy/kubernetes/GUIDE.md).
- **Earlier emulator evaluation:** [historical technical report](docs/TECHNICAL_REPORT.md).

## Scope and current checkpoint

The long-term objective is to serve a required workload with acceptable response times using GPU resources and energy efficiently, and to determine when replicas should be maintained, added, removed or placed elsewhere.

| Area | Current implementation | Remaining work |
| --- | --- | --- |
| Routing | OpenAI-compatible JSON/SSE proxy; Round robin, Capacity and KubeAI policies; health/model checks; atomic local reservations and failure cooldown | Calibrate estimates and weights; isolate the contribution of each signal |
| Experiments | Closed-loop workers, seeded arrival traces, raw requests, manifests, streaming TTFT and repeated comparisons | Broader loads, seeds, models, saturation tests and controlled single-versus-multiple-replica experiments |
| Energy | Optional GPU utilization, VRAM and board-power samples; coverage-aware gross energy estimates | Independent validation and remote/multi-GPU measurement |
| Dashboard | Endpoint editing, connection tests, one background benchmark at a time, saved comparisons and six interactive charts | Further research views as needed; no overall evaluation score is implemented |
| Placement | First-fit/best-fit recommendations and optional Kubernetes Pod generation | Live placement validation and coordination with changing allocations |
| Replica lifecycle | Manually deployed replicas; discovery support and deployment templates | Automatic replica-count decisions, scale-in/scale-out and GPU provisioning |

**Checkpoint — 15 November 2026:** a local hardware campaign compared all three policies with three repeats each, six workers, 30-second measurement targets and two vLLM replicas sharing one RTX 4070 Laptop GPU. All 5,589 recorded requests completed at the protocol level, with no request failures or client drops. These are **exploratory** results: model revision and serving-image provenance were not recorded in those run manifests. See the [progress report](docs/PROGRESS.md) for metrics, raw-artifact links and limitations.

The current code checks comprise **128 unit/integration tests** and a JavaScript syntax check. Current evidence does not establish general superiority, production readiness, fewer required GPUs or automatic scaling.

## How routing works

The router first filters for fresh healthy endpoints that advertise the requested model, allow it in configuration and support the requested output budget. Full prompt/context validation remains the model server's responsibility.

| Policy | Selection rule |
| --- | --- |
| `round_robin` | Rotate across eligible replicas |
| `capacity` | Minimize local `(inflight + 1) / capacity` |
| `kubeai` | Minimize a weighted load/service, KV-cache and optional physical-memory score |

KubeAI uses:

```text
load    = max(local inflight, observed waiting + running) / configured capacity
service = configured prefill_seconds + requested max_tokens × token_seconds

score = queue_weight × load × service
      + service_weight × service
      + kv_weight × KV pressure
      + memory_weight × observed physical memory pressure
```

Lower scores are preferred; ties rotate. The estimates and weights are manual research parameters, not learned or calibrated automatically. `capacity` is a relative serving estimate, not a GPU percentage or an enforced request limit.

vLLM queue/running/KV signals are polled directly. Stale signals are omitted and fresh local reservations remain usable. Physical memory participates only when a replica has a valid, fresh `gpu_metrics_url`; this mapping is blank in the current shared-GPU configuration. GPU utilization, watts and energy are **measured for evaluation, not inputs to the current routing score**.

## Prerequisites

Choose the CPU-only setup or the real-vLLM setup; both use the same router and dashboard.

| Requirement | Needed for |
| --- | --- |
| Python 3.11+ | Router, emulators, benchmark and placement CLI; no third-party Python runtime dependencies |
| Node.js and Make | The complete `make test` check; the dashboard itself needs no Node build |
| Docker Engine and Compose plugin | Container examples and the supplied vLLM setup |
| Working NVIDIA driver and NVIDIA Container Toolkit | GPU access inside vLLM containers |
| Network access and disk space | Downloading container images and model weights on a fresh machine |

For installation details, use the official [Python virtual-environment guide](https://docs.python.org/3.11/library/venv.html), [Docker Compose installation guide](https://docs.docker.com/compose/install/) and [NVIDIA Container Toolkit guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Copy or check out the project, replace the path below with its directory, and prepare Python:

```bash
cd /path/to/kubeai
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m kubeai --help
```

Run commands from the project root. In every new terminal, activate the same `.venv`. No `pip install`, frontend build or GPU is needed for the CPU-only path.

## Setup A: local experiments without a GPU

Use three terminals from the project root.

**Terminal 1 — emulator A:**

```bash
python3 -m kubeai emulator --port 8001 --capacity 2 --token-seconds 0.004
```

**Terminal 2 — emulator B:**

```bash
python3 -m kubeai emulator --port 8002 --capacity 2 --token-seconds 0.001
```

**Terminal 3 — router and dashboard:**

```bash
python3 -m kubeai router --config config/local.json --results results --port 8080
```

Open **http://127.0.0.1:8080**. Two healthy replicas should appear. Verify the model list or send a request:

```bash
curl -fsS http://127.0.0.1:8080/v1/models

curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Explain GPU inference."}],"max_tokens":16,"stream":true,"stream_options":{"include_usage":true}}'
```

Choose the **Emulator** environment label for these servers. Emulators return synthetic text and timing; their metrics do not establish model quality or hardware efficiency. Stop terminal-launched services with Ctrl+C.

For an automated comparison using disposable CPU HTTP servers:

```bash
python3 -m kubeai demo --repeats 3 --requests 48 --output results/demo
```

This runs three scenarios × three policies × three repeats with rotated policy order, saves `REPORT.md` and raw artifacts, and closes its temporary servers. Point a dashboard at the directory afterward:

```bash
python3 -m kubeai router --config config/local.json --results results/demo --port 8080
```

The persistent emulators above provide live health; the completed demo's temporary servers are already stopped.

## Setup B: real vLLM replicas

Use this instead of the emulators when ports 8001 and 8002 are free. If the two vLLM containers already exist and are healthy, use them and skip their creation.

### Check the GPU runtime

The following commands check local prerequisites:

```bash
nvidia-smi
docker --version
docker compose version
```

The local reference uses these pinned artifacts:

```bash
export VLLM_IMAGE='vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3'
export MODEL_REVISION='c1899de289a04d12100db370d81485cdf75e47ca'
```

Check container GPU access with the selected image:

```bash
docker run --rm --gpus all --entrypoint nvidia-smi "$VLLM_IMAGE"
```

These pins are reference setup values, not retrospective proof of the artifacts used by historical runs whose manifests say `unknown`.

### Start two replicas on one GPU

This profile was exercised locally with `Qwen/Qwen3-0.6B`, a 2,048-token model context and an 8,188 MiB RTX 4070 Laptop GPU. Other models/devices can require different memory budgets.

Start replica A:

```bash
docker compose -p kubeai-shared -f compose.gpu.yaml run -d --no-deps \
  --name kubeai-vllm-a --publish 127.0.0.1:8001:8000 vllm \
  --model Qwen/Qwen3-0.6B --revision "$MODEL_REVISION" \
  --max-model-len 2048 --gpu-memory-utilization 0.40
```

Wait for readiness; inspect logs while it starts:

```bash
docker logs -f kubeai-vllm-a
```

Ctrl+C exits log following. Once the model endpoint responds, create replica B:

```bash
curl -fsS http://127.0.0.1:8001/v1/models

docker compose -p kubeai-shared -f compose.gpu.yaml run -d --no-deps \
  --name kubeai-vllm-b --publish 127.0.0.1:8002:8000 vllm \
  --model Qwen/Qwen3-0.6B --revision "$MODEL_REVISION" \
  --max-model-len 2048 --gpu-memory-utilization 0.40

docker logs -f kubeai-vllm-b
```

After B starts, check it and launch the host router:

```bash
curl -fsS http://127.0.0.1:8002/v1/models
python3 -m kubeai router --config config/vllm-shared.json --results results --port 8080
```

Open **http://127.0.0.1:8080** and verify both replicas. The two `0.40` settings are per-instance **VRAM budgets**, not 40% compute isolation, an 80% GPU-utilization cap or two independent GPUs.

`compose.gpu.yaml` also contains a separate single-server profile on port 8000 using a `0.75` memory budget. `docker compose -f compose.gpu.yaml up -d --wait` starts that profile and Prometheus; it is not the shared-GPU launch sequence above.

### Connect existing local or cloud servers

The router only needs reachable OpenAI-compatible serving endpoints. In **Endpoint configuration**, enter each server's base URL without `/v1`, name and allowed model IDs; use **Test connection**, then **Apply endpoints**. This does not deploy a server or change its GPU allocation.

Credentials are supplied through server-side environment variables named by `api_key_env`; the UI accepts the variable name, not the secret. Restart the router after changing its environment. Optional `KUBEAI_API_KEY` protects router APIs; the page keeps its bearer token in memory.

## Run benchmarks from the dashboard

1. Use a healthy dedicated fleet and apply/discard endpoint drafts first.
2. Select **Emulator** or **Hardware** to describe the actual endpoints. The label does not switch servers.
3. Choose a model and one routing policy.
4. Choose **Closed-loop workers** or **Saved arrival trace**.
5. Set warm-up and, for hardware, fill declared model revision and serving-image digest. For the pinned setup, use the values of `MODEL_REVISION` and `VLLM_IMAGE` above.
6. Optionally enable **Sample NVIDIA GPU utilization, VRAM and power on this router host**.
7. Select **Run benchmark** and wait for the saved run path.

**Each click runs one policy.** The job temporarily selects it, then restores the configured policy. It does not automatically run all three. Endpoint writes and additional dashboard jobs are blocked during the job; unrelated inference/CLI clients can still add load. Browser closure does not stop a job; router restart interrupts it. External configuration edits invalidate the active experiment.

Closed-loop controls support 1–128 workers, durations up to 300 seconds and seeds. Saved traces must be JSONL files under the router's `results/workloads/` directory; dashboard limits are 8 MiB, 20,000 requests, 600 seconds and 1–128 client in-flight requests. Warm-up is per requested model through the router; for a controlled campaign also warm every physical replica directly.

For example, generate a reusable trace:

```bash
python3 -m kubeai trace --output results/workloads/steady-r20-seed42.jsonl \
  --requests 200 --rate 20 --scenario steady --seed 42
```

For repeated policy comparisons, keep the serving configuration, workload, warm-up, sampling, provenance and benchmark settings fixed. Rotate order:

| Round | Policy order |
| --- | --- |
| 1 | Round robin → Capacity → KubeAI |
| 2 | Capacity → KubeAI → Round robin |
| 3 | KubeAI → Round robin → Capacity |

## Read saved comparisons

Results stay under `results/dashboard-benchmarks/JOB_ID/RUN_ID/`; new runs do not overwrite old runs.

- **Recorded provenance:** group runs with matching required context, including declared model/image provenance for hardware.
- **Exploratory comparison:** explicitly allow missing model revision/image digest only. Other missing context, invalid runs and conflicting known provenance remain separated or excluded.
- **Individual runs — no averaging:** retain each observation separately.
- **View latest exploratory setup:** open the most recent eligible group allowed by the environment/scenario filters.

A selected setup has six charts: P95, TTFT P95, completed req/s, mean GPU watts, J/completed request and J/1000 output tokens. Bars are the existing per-policy means; dots are individual runs. Hover/focus a dot for its value and select it to inspect that run. Axes start at zero and colors stay consistent. Counts of completions, failures, drops and token-limit responses remain visible; **Show detailed measurements** opens the tables.

A selected group keeps its own results when a different setup is run. Matching repeats update that group's means and counts. The viewer considers the latest 200 summaries, including nested directories.

**Inspect** includes recorded context, per-replica request shares/latency/failures and physical GPU samples. Missing measurements stay unavailable. Per-replica CSV inspection is limited to 64 MiB and 200,000 rows; requests without replica metadata stay separate. Request shares do not attribute GPU energy to replicas.

## CLI experiments and recorded artifacts

A hardware closed-loop example against the shared-GPU router:

```bash
python3 -m kubeai benchmark --url http://127.0.0.1:8080 \
  --output results/cli-benchmarks --duration 30 --concurrency 6 --seed 42 --warmup 2 \
  --environment hardware --policy kubeai --gpu \
  --model-revision "$MODEL_REVISION" --image-digest "$VLLM_IMAGE" \
  --metrics-url http://127.0.0.1:8080/metrics \
  --metrics-url http://127.0.0.1:8001/metrics \
  --metrics-url http://127.0.0.1:8002/metrics
```

`--policy` labels/checks the observed policy; it **does not reconfigure the router**. For CLI comparisons, change the JSON configuration's `policy` and verify `/api/state` before each run. Valid file changes reload during polling. The dashboard runner handles its temporary policy selection itself.

Replay the same saved external arrivals instead of a worker loop:

```bash
python3 -m kubeai benchmark --url http://127.0.0.1:8080 \
  --output results/cli-benchmarks --trace results/workloads/steady-r20-seed42.jsonl \
  --environment hardware --policy kubeai --gpu \
  --model-revision "$MODEL_REVISION" --image-digest "$VLLM_IMAGE"
```

For CPU emulators, use `--environment emulator` and omit GPU sampling/hardware provenance. `--request bench/request.json` selects a fixed payload for closed-loop tests; it cannot be combined with `--trace`.

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Settings, timing window, source/input hashes, provenance and router snapshots |
| `requests.csv` | Every offered request, including failures and client drops |
| `inputs.jsonl` / `trace.jsonl` | Exact input pool / fixed-arrival trace |
| `telemetry.jsonl` | Raw metrics, failures and optional timestamped NVIDIA samples |
| `summary.json` | Per-run statistics, outcomes and optional `gpu_energy` |
| `job.json` | Dashboard execution state, in the parent job directory |
| `prometheus.json` | Optional recorded-interval query results when `--prometheus-url` is supplied |

`results/*` is ignored by version control by default. Historical local artifacts referenced in the progress report may not accompany a fresh checkout; archive the full run directories when sharing evidence.

## Measurement rules and limits

- P50/P95 use completed responses and nearest-rank percentiles. Grouped P95 is a mean of per-run P95 values, not a percentile pooled across requests. Dots/ranges are observations, not confidence intervals.
- Completion requires a terminal JSON/SSE response with `stop`, `length` or `tool_calls`. Token-limited responses are counted separately; completion does not establish answer quality.
- TTFT measures the first generated content/reasoning/tool delta, excluding role-only events. Non-streaming runs have no client TTFT.
- Throughput includes request drain. Scheduled latency additionally includes dispatch lag. Fixed-arrival requests that hit `--max-inflight` are recorded as drops, not silently omitted; drops are also part of failures.
- Warm-up is excluded. Closed-loop concurrency and externally scheduled arrival rates answer different questions; the same input pool does not mean identical offered-request counts.
- GPU sampling is opt-in and local to the **benchmark/router host**, not remote endpoint GPUs. Gross energy includes idle/other GPU activity and excludes system CPU/RAM.
- Energy integrates board watts over the measured monotonic-clock window, with initial/final boundary samples and no extrapolation. Pairs farther apart than `max(3 × sampling interval, 1 second)` are not integrated; missing readings break continuity.
- Full-window energy/mean power require complete coverage. Partial covered energy stays explicit. J/request requires completed requests; J/1000 tokens also requires positive complete output-token usage. Missing data never becomes zero.
- Physical devices count once per UUID, even when replicas share one GPU. Sampled power/utilization maxima are not continuous peaks.
- Interrupted jobs, unstable configurations and policy/hash inconsistencies are excluded from comparisons. Source, workload, worker/duration, endpoint or sampling changes create separate groups.
- A declared image/revision is recorded provenance, not independent deployment attestation. Serving flags, thermal conditions, other traffic and per-replica warm-up still need control.

## Optional monitoring, containers and placement

### Prometheus and the memory exporter

The host dashboard works without Prometheus or Grafana. Prometheus configuration is supplied for additional observation; Grafana is not part of the current application.

`monitoring/prometheus.hardware.yml` targets the separate server on 8000 by default. To monitor the shared-GPU setup, edit its vLLM targets to 8001 and 8002 before using that profile. Its GPU-exporter target is 9401.

The optional exporter exposes one physical device's memory/utilization:

```bash
python3 -m kubeai gpu-exporter --gpu-id 0
```

To use physical memory in routing, map `gpu_metrics_url` to that exporter for replicas using the same device. It does not make watts/energy routing inputs. In the current shared-GPU configuration the exporter mapping is empty.

### CPU Docker Compose profile

```bash
docker compose build
docker compose up -d --wait
```

This starts CPU emulators, router/dashboard on 8080 and Prometheus on 9090. Use it instead of host services on those ports. The supplied router mounts config/results **read-only**, so dashboard endpoint editing/jobs need a different writable setup; the host-router path above supports them. Host CLI experiments can still write results for the container dashboard to read.

```bash
docker compose down
```

The named Prometheus volume is retained. For an isolated full-stack verification with temporary Compose resources, use `python3 scripts/verify_containers.py`; it requires Compose 2.24.4+.

### Placement and Kubernetes

Compare recommendation rules on the supplied synthetic inventory:

```bash
python3 -m kubeai placement --inventory config/inventory.example.json \
  --request config/placement-request.json --at 1000 --policy first_fit

python3 -m kubeai placement --inventory config/inventory.example.json \
  --request config/placement-request.json --at 1000 --policy best_fit
```

For live recommendations, use fresh inventory with real node names, whole-GPU allocations and per-device memory, plus CPU/RAM, model compatibility and schedulability. Omit `--at`; the default freshness limit is 30 seconds.

`--pod-output` can generate a Pod when supplied with a verified image digest and model revision. The CLI never applies a cluster change or reserves live capacity. Automatic scale-out, scale-in, node provisioning and multi-GPU efficiency remain future work.

See [deploy/kubernetes/GUIDE.md](deploy/kubernetes/GUIDE.md) for discovery, deployment templates, persistent Prometheus and GPU-serving prerequisites. Live Kubernetes rollout was not validated in the recorded work.

## API overview

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions`, `GET /v1/models` | JSON/SSE inference and verified model listing |
| `GET /healthz`, `GET /readyz`, `GET /metrics` | Liveness, readiness and Prometheus metrics |
| `GET /api/state`, `GET /api/runs` | Live configuration/state and saved summaries |
| `GET /api/runs/RUN_ID` | On-demand per-replica request details |
| `GET /api/comparisons` | Comparisons with required recorded provenance |
| `GET /api/comparisons?mode=exploratory` | Explicit incomplete-provenance comparisons |
| `GET /api/endpoints`, `POST /api/endpoints/test`, `PUT /api/endpoints` | Inspect, probe and save endpoint configurations |
| `GET /api/benchmark/options`, `GET /api/benchmark`, `POST /api/benchmark` | Inputs, latest job state and one-job execution |

Operator writes/probes require the configured bearer token or, without a key, loopback clients/hosts and same-origin JSON. Static config is required for endpoint editing and dashboard execution; Kubernetes discovery is read-only for these controls. Use one router process for experiments because reservations are process-local. This is a local research service, not a production-hardened multi-tenant gateway.

## Repository map and checks

```text
kubeai/                    Router, policies, workloads, measurements, placement and dashboard assets
config/                    Local, shared-GPU, Docker and synthetic-inventory examples
compose.yaml               CPU emulator/container profile
compose.gpu.yaml           Pinned single-vLLM profile; reusable for explicit shared-GPU runs
monitoring/                Prometheus scrape configurations
deploy/kubernetes/         Cluster templates and GUIDE.md
results/                   Local generated artifacts; ignored by default
bench/                     Preserved early baseline scripts, payload and CSV
docs/PROGRESS.md            Current milestones, hardware evidence and next steps
docs/TECHNICAL_REPORT.md    Historical emulator-release report
```

Run the complete existing checks:

```bash
make test
```

The suite uses temporary CPU HTTP emulators and mocked NVIDIA readings, not the deployed GPU services. Loopback access is required. If Make is unavailable, its two checks are:

```bash
python3 -m unittest discover -s tests -v
node --check kubeai/static/app.js
```

CI checks Python 3.11–3.13 and saves a small emulator comparison artifact. For the next research priorities and the boundary between completed work and proposals, use [the progress report](docs/PROGRESS.md).

