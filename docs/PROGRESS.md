# KubeAI — progress and evidence

Snapshot: **15 November 2026**. This document records completed work, the evidence behind it and the next research steps. Setup instructions and project scope live in the single [main README](../README.md).

## Where the project stands

KubeAI is a working research prototype with real-model routing and local GPU measurements. The current stage is **experimental validation of routing and measurement**. A production orchestration service, automatic replica lifecycle and multi-GPU efficiency have not been demonstrated.

| Workstream | Completed work | Verification boundary |
| --- | --- | --- |
| Routing | Round robin, Capacity and KubeAI; model/health filtering, local reservations, cooldown and config reload | Unit/integration checks, emulator experiments and local vLLM campaigns |
| Observation | vLLM queue/running/KV polling; Prometheus exposition; optional single-device memory exporter | Current shared-GPU config does not connect physical memory to routing |
| Benchmark engine | Seeded arrival traces, closed-loop workers, streaming TTFT, per-request CSV, input/source hashes and manifests | Runtime/serving conditions still need operator control |
| Dashboard controls | Endpoint drafts/probes, versioned persistence, local/cloud URLs and one background benchmark job at a time | Model deployment and GPU allocation remain external |
| Saved analysis | Matching-context groups, explicit incomplete-provenance comparisons, per-replica request breakdown and missing-data handling | Exploratory groups are not independently attested deployments |
| Visual comparison | Six charts with policy means/per-run dots; detailed tables remain available; outcome counts stay visible | Data/unit checks completed; browser appearance reviewed through the local UI |
| GPU energy | Board-power samples, boundary-aware integration, per-UUID totals, coverage and normalized energy | Gross local GPU estimates, not system-wide or per-replica energy |
| Placement | First-fit/best-fit recommendation and optional Pod manifest generation | Synthetic inventory checks; no automatic cluster writes or live reservations |
| Deployment | Docker/Compose profiles and Kubernetes discovery/manifests | Historical container verification and local rendering; no recorded live Kubernetes GPU rollout |
| Checks | 128 unit/integration tests and JavaScript syntax check passed | Socket tests use temporary CPU emulators; NVIDIA access is mocked |
| Scientific communication | Poster framing/text aligned with the research question and target architecture | Final publication figures and architecture artwork remain to be prepared |

The latest dashboard chart work also checked **18 policy means and 54 individual values** against the nine-run hardware dataset. Millisecond/energy units, missing-versus-zero behavior and HTML/JavaScript element references were checked. The chart work did not launch GPU inference.

## What KubeAI currently decides

The implemented routing score uses local/observed load, configured service-time estimates, KV-cache pressure and optional physical memory pressure. It selects the lowest score among fresh, healthy, compatible replicas.

- Capacity and service coefficients are configured research estimates; they are not learned or calibrated automatically.
- Physical memory requires a fresh, correctly mapped exporter. The current shared-GPU config leaves that mapping empty.
- GPU utilization, power and energy are evaluation measurements, not inputs to the current routing score.
- The placement CLI recommends a compatible location for a requested allocation; it does not decide replica count, apply Pods or provision GPUs.
- An overall algorithm evaluation score and automatic winner selection are not implemented.

See [policy.py](../kubeai/policy.py), [placement.py](../kubeai/placement.py) and the [shared-GPU configuration](../config/vllm-shared.json).

## Completed experimental stages

1. Preserved the early fixed-prompt single-vLLM baseline scripts and CSV under [bench/](../bench/).
2. Implemented the routing/measurement prototype and compared policies through disposable HTTP emulators. The supplied release evaluation contains 27 runs and 1,296 completed requests; its data is synthetic. See the [emulator report](../results/release-evaluation/REPORT.md) and [historical technical report](TECHNICAL_REPORT.md).
3. Recorded local single-server concurrency and fixed-arrival measurements, then three shared-GPU fixed-arrival pilots under `results/hardware-*/`. Those pilots are separate from the newer closed-loop campaign below.
4. Added real endpoint administration, dashboard-triggered individual benchmarks and saved context-aware comparisons.
5. Added per-replica analysis and local GPU power/energy estimates; validated calculation behavior with known synthetic power curves and CPU-only integration tests.
6. Completed the nine-run hardware campaign below with policy-order rotation.
7. Added six visual comparisons from the existing saved data without rerunning the hardware campaign.

Earlier single-server and shared-GPU pilots differ in topology and/or measurement path. They do not isolate the effect of adding a second replica. Emulator, direct-server, routed and different-workload results must remain separate.

## Hardware campaign: three policies × three repeats

### Recorded protocol

| Item | Value |
| --- | --- |
| Model identifier | `Qwen/Qwen3-0.6B` |
| Hardware | One NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB |
| Serving topology | Two local vLLM replicas sharing that physical device |
| Working-session serving profile | 2,048-token context and `0.40` VRAM budget per replica; the budgets do not isolate compute |
| Workload mode | Closed-loop, 6 workers, 30-second request-issuance target; final drain is included |
| Inputs | Same saved varied-input pool, seed 42; output budgets of 8/16/32/64 tokens |
| Warm-up | 2 requests per requested model through the router, excluded from measurement |
| Streaming | Enabled |
| GPU sampling | Enabled, 1-second polling wait; complete integration coverage in every run |
| Model/image provenance | `unknown` in all nine manifests |
| Classification | **Exploratory comparison** |

The serving flags above describe the setup used in the working session. Image/model provenance was not independently captured in these run manifests. Reference pins in the setup guide must not be substituted retrospectively for the missing declarations.

Policy order was rotated:

| Round | Order |
| --- | --- |
| 1 | Round robin → Capacity → KubeAI |
| 2 | Capacity → KubeAI → Round robin |
| 3 | KubeAI → Round robin → Capacity |

The input pool was identical; the number of offered/completed requests was not fixed because this was a closed-loop throughput experiment.

### Observed results

Values below are **means of per-run measurements**, including a mean of each run's P95 rather than a pooled request percentile.

| Policy | Runs | Completed req/s | P95 (ms) | TTFT P95 (ms) | Mean GPU power (W) | J/completed request | J/1000 output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Round robin | 3 | 18.23 | 814.3 | 42.5 | 71.25 | 3.91 | 158.76 |
| Capacity | 3 | 18.26 | 813.6 | 37.7 | 71.18 | 3.90 | 158.29 |
| KubeAI | 3 | 24.76 | 738.2 | 40.6 | 72.87 | 2.95 | 120.51 |

- **5,589/5,589** recorded requests completed at the protocol level; no request failures or client drops.
- **4,960** completions reached their output-token limit. They remain distinct from transport failures; answer quality was not evaluated.
- All nine runs have `gpu_energy.status = estimated` and 100% integration-window coverage under the recorded gap rule.
- Observed per-run P95 ranges: Round robin **813.5–814.8 ms**, Capacity **813.4–813.8 ms**, KubeAI **679.6–782.8 ms**. These ranges are not confidence intervals.

Relative to Round robin, KubeAI's observed mean throughput was about **35.8% higher**, mean per-run P95 **9.3% lower**, and mean J/completed request **24.6% lower**. Capacity had the lowest TTFT P95.

Mean total energy per run was approximately **2,176 J** for Round robin, **2,167 J** for Capacity and **2,209 J** for KubeAI. KubeAI used about **1.5% more total energy** than Round robin while completing more work. The observation supports better energy efficiency per unit of delivered work in this configuration, not reduced total watts/joules or fewer GPUs.

### Evidence locations

Run identifiers and manifest wall-clock timestamps use UTC. Each link opens a raw summary; its directory also contains the manifest, request CSV, exact inputs and raw telemetry.

| Round | Policy | Saved summary |
| --- | --- | --- |
| 1 | Round robin | [20261115T015931-round_robin-4c8f1f6f](../results/dashboard-benchmarks/20261115T015931-bb36abaf/20261115T015931-round_robin-4c8f1f6f/summary.json) |
| 1 | Capacity | [20261115T020008-capacity-e8ddbf8b](../results/dashboard-benchmarks/20261115T020008-9b651345/20261115T020008-capacity-e8ddbf8b/summary.json) |
| 1 | KubeAI | [20261115T020043-kubeai-92036471](../results/dashboard-benchmarks/20261115T020043-16387707/20261115T020043-kubeai-92036471/summary.json) |
| 2 | Capacity | [20261115T020848-capacity-4f1421a2](../results/dashboard-benchmarks/20261115T020848-9191570e/20261115T020848-capacity-4f1421a2/summary.json) |
| 2 | KubeAI | [20261115T021242-kubeai-0e1a55ed](../results/dashboard-benchmarks/20261115T021242-0a808b77/20261115T021242-kubeai-0e1a55ed/summary.json) |
| 2 | Round robin | [20261115T021626-round_robin-b4d9f42d](../results/dashboard-benchmarks/20261115T021626-d27f98b8/20261115T021626-round_robin-b4d9f42d/summary.json) |
| 3 | KubeAI | [20261115T022742-kubeai-0b8f089d](../results/dashboard-benchmarks/20261115T022742-ca759b11/20261115T022742-kubeai-0b8f089d/summary.json) |
| 3 | Round robin | [20261115T022911-round_robin-ca73db43](../results/dashboard-benchmarks/20261115T022911-c67dfd75/20261115T022911-round_robin-ca73db43/summary.json) |
| 3 | Capacity | [20261115T023048-capacity-cacc49d7](../results/dashboard-benchmarks/20261115T023048-f365686b/20261115T023048-capacity-cacc49d7/summary.json) |

Recorded benchmark-source SHA256:

```text
c61092d3281e7679f5701f3cc6b083a404fe562f26a45d5df7faedd233a66bcd
```

Recorded input-pool SHA256, stored as `trace_sha256` even for closed-loop runs:

```text
651093ad6a2acf247b753b1217cf2a828e55a921d9bfd4bc68a52db55f6ef873
```

The viewer's current grouping ID begins `6d82d7fc`. Later source/UI changes can create a new benchmark-source hash and separate future runs. Historical artifacts are not rewritten to force grouping.

Raw `results/` directories are local and ignored by version control by default. A fresh checkout may lack these linked artifacts; retain/export the complete run directories with the report for reproducibility.

### Interpretation limits

This campaign covers one model, one input seed/pool, one concurrency/duration profile and one shared physical GPU. Three repeats show observed variation, not statistical significance across workloads or a causal explanation of which score term helped.

Additional limits:

- Model revision/image digest are missing from the manifests, and serving flags were not independently attested.
- GPU-board energy includes idle and unrelated GPU activity; it excludes host CPU/RAM and is not attributed to individual replicas.
- Sensor sampling/integration is an estimate. Full integration coverage does not mean continuous exact power metering.
- Replica warm-up through the router does not guarantee every physical endpoint was warmed identically.
- KV/cache, batching, clocks and thermal conditions may affect results; their causal contribution was not isolated.
- Successful completion includes token-limited output. Semantic answer quality was not measured.
- There is no controlled current single-versus-two-replica comparison, live multi-GPU result, measured cloud cost reduction or validated autoscaling decision.

## Next steps and acceptance criteria

Work through these one experimental step at a time:

1. **Freeze the serving protocol.** Record actual model/image revisions, serving flags, device/driver identity, input hashes and per-replica warm-up. Preserve historical missing metadata instead of inventing it.
2. **Define quality-of-service criteria before the next campaign.** Agree acceptable P95/TTFT, failure/drop rates and how output truncation is interpreted. No production SLO has been selected.
3. **Measure the operating frontier.** Replay fixed-arrival steady/burst workloads across rates, seeds and lengths; retain failure/queue/latency/energy behavior near saturation.
4. **Calibrate and isolate signals.** Measure service/capacity estimates and compare controlled KubeAI variants to identify the contribution of load, KV and optional physical-memory terms.
5. **Evaluate replica count and placement.** Compare one/two replicas under a controlled resource envelope and the same router path; then validate independent GPUs or a cloud/Kubernetes testbed.
6. **Add lifecycle decisions after measurement.** Start with observable recommendations for maintaining/adding/removing replicas, with fresh capacity checks, cooldown and hysteresis; validate before automatic actuation.
7. **Define evaluation scoring only after its criteria are explicit.** Keep any future score versioned, transparent and conditional on complete comparable measurements. Scoring is currently deferred.
8. **Finish communication artifacts.** Export readable preliminary performance/energy figures and a target-architecture diagram for the scientific poster.

Success is a reproducible prototype and a defensible account of where a policy helps, where it does not and why. No fixed improvement percentage is promised outside the measured configuration.

## Documentation consolidation

The project concept, current scope and setup are now consolidated in [README.md](../README.md). The stale root duplicate has been retired. The Kubernetes instructions remain in [GUIDE.md](../deploy/kubernetes/GUIDE.md), and the earlier technical report is retained as a dated historical snapshot. Original benchmark inputs/results and all hardware run artifacts are preserved.

