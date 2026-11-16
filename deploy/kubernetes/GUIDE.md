# Kubernetes deployment guide

For current scope and local setup, use the [main README](../../README.md). Current implementation progress and hardware evidence are recorded in [PROGRESS.md](../../docs/PROGRESS.md).

This deploys two CPU-only emulator Pods, a router that discovers their individual ready addresses through EndpointSlices, and Prometheus with a PVC. No GPU is required for this path. A running Kubernetes cluster with a default storage class is required; otherwise configure the PVC storage class explicitly.

## Build, load and deploy

```bash
docker build -t kubeai:local .
```

For kind, load the image into your chosen cluster:

```bash
kind load docker-image kubeai:local
```

For minikube, use `minikube image load kubeai:local`. For a remote cluster, push to your registry and change the emulator/router image references to the resulting immutable digest. These are development manifests; record the final rendered manifest and image identity for experiments.

Render and inspect before applying:

```bash
kubectl kustomize deploy/kubernetes --output results/kubernetes.rendered.yaml
kubectl apply -k deploy/kubernetes
kubectl -n kubeai rollout status deployment/kubeai-emulator
kubectl -n kubeai rollout status deployment/kubeai-router
kubectl -n kubeai rollout status deployment/kubeai-prometheus
```

In separate terminals, forward the local ports:

```bash
kubectl -n kubeai port-forward service/kubeai-router 8080:8080
kubectl -n kubeai port-forward service/kubeai-prometheus 9090:9090
```

Open http://127.0.0.1:8080. Run host benchmarks against the forwarded router, then keep the host results dashboard pointed at the generated results directory. The in-cluster dashboard has live state; benchmark artifacts remain on the load-generator host. To view saved host experiments, run a host router on port 8081 with a config pointing to the forwarded endpoints, or use the supplied local emulator config for a results-only view:

```bash
python3 -m kubeai router --port 8081 --results results
```

## Discovery and changes

The router's namespaced Role grants only `list` on EndpointSlices. It reads the projected service-account token on each poll, verifies the Kubernetes API's TLS certificate, filters the `kubeai-backends` service by label, and accepts ready, non-terminating endpoints with the named `http` port. It then verifies `/v1/models` and `/metrics` on each individual address. Duplicate slice entries are deduplicated; IPv4/IPv6 addresses are supported.

```bash
kubectl -n kubeai scale deployment/kubeai-emulator --replicas=3
kubectl -n kubeai get endpointslices -l kubernetes.io/service-name=kubeai-backends
```

The dashboard should show the new replica after discovery and a successful health probe. On discovery failures the last set is retained only until its configured freshness deadline; after that discovered endpoints are removed. In-flight requests retain their original connection.

Edit `router.json` and apply the Kustomization to change policy or defaults. ConfigMap volume projection is eventually consistent; the router reloads the updated file without restarting. The example has one router Pod because local reservations do not coordinate across processes. All discovered replicas in this simple service share the same configured capacity/service estimates; use separate static entries when comparing heterogeneous hardware or multiple model pools.

Prometheus uses a headless service to scrape each emulator independently. It retains blocks for two days or 512 MB, with a 2 GiB PVC allowing space for WAL and compaction. Use Linux-compatible storage. The base Python image and Prometheus image are pinned by digest.

## GPU-serving Pods

The GPU path requires a functional NVIDIA driver, container runtime integration and the NVIDIA Kubernetes device plugin. Kubernetes requires equal GPU requests and limits when both are specified. See the official [GPU scheduling documentation](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/).

Collect a fresh inventory in the schema shown in `config/inventory.example.json`. Count only unallocated whole devices, account for existing CPU/RAM requests, record per-device free/total memory and observation timestamps, and exclude nodes with taints the generated Pod cannot tolerate. Use the actual Kubernetes node metadata names. The example timestamps and device names are synthetic and only usable with `--at 1000` for offline demonstration.

```bash
python3 -m kubeai placement --inventory results/live-inventory.json \
  --request config/placement-request.json --policy best_fit \
  --image 'vllm/vllm-openai@sha256:YOUR_VERIFIED_DIGEST' \
  --model-revision YOUR_VERIFIED_REVISION --pod-output results/model-pod.json
```

Inspect the generated Pod and apply it with `kubectl apply -f results/model-pod.json` when ready. The chosen node is enforced through required affinity; the normal scheduler still checks resource availability. A conflicting concurrent allocation can leave the Pod Pending rather than overcommit a whole-GPU request. This policy does not reserve live capacity or act as a custom Kubernetes scheduler. Refresh the inventory and re-evaluate if placement is delayed.

For routing, create a service selecting `app: kubeai-vllm` with a port named `http` targeting 8000. Change the router discovery service to that service, remove the emulator Deployment if no longer needed, and set capacity/timing defaults from measurements. Create separate model pools/static endpoints for multiple served models. Update Prometheus targets and record model/image revisions, serving flags and hardware metadata.

Never create two 75%-memory instances on the same 8 GB GPU to claim a two-replica hardware comparison. The supplied placement rule requests whole devices. Multi-GPU experiments need independent devices with comparable serving conditions.

## Verification boundary

The manifests were rendered locally with `kubectl kustomize`. The container application and Prometheus were exercised through Docker Compose. No live Kubernetes deployment or GPU Pod was applied during this implementation, and a cluster rollout has not been claimed.
