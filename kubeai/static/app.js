"use strict";
const $ = id => document.getElementById(id);
let runs = [];
let comparisonGroups = [];
let comparisonIssue = "";
let comparisonCatalog = {recorded: [], exploratory: []};
let comparisonIssues = {recorded: "", exploratory: ""};
let comparisonsReady = false;
let pickLatestComparison = false;
let detailVersion = 0;
let loading = false;
let endpointRows = [];
let endpointRevision = null;
let endpointEditable = false;
let endpointDirty = false;
let endpointBusy = false;
let endpointLoading = false;
let endpointConflict = false;
let endpointSequence = 0;
let benchmarkActive = false;
let benchmarkSubmitting = false;
let benchmarkOptions = null;
let benchmarkOptionsLoading = false;
let benchmarkStatusLoading = false;
let benchmarkStatusVersion = 0;
let benchmarkLastSignature = "";
let benchmarkModelsSignature = "";
const display = value => value == null ? "—" : Number(value).toFixed(3);
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function apiHeaders(json = false) {
  const headers = $("api-key").value ? {Authorization: "Bearer " + $("api-key").value} : {};
  if (json) headers["Content-Type"] = "application/json";
  return headers;
}
async function endpointApi(path, method = "GET", body) {
  const response = await fetch(path, {method, headers: apiHeaders(body !== undefined), cache: "no-store",
    ...(body === undefined ? {} : {body: JSON.stringify(body)})});
  let value;
  try { value = await response.json(); } catch { throw new Error("The endpoint API returned an unreadable response."); }
  if (!response.ok) {
    if (response.status === 404 && path.startsWith("/api/runs/")) throw new Error("Saved request details are unavailable. Refresh the run list or restart an older router.");
    if (response.status === 404) throw new Error(path.startsWith("/api/benchmark") ?
      "Restart the router to enable dashboard benchmarks." : "Restart the router to enable endpoint management.");
    const error = new Error(value.error?.message || `Router returned HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return value;
}
function endpointStatus(message, error = false) {
  $("endpoint-status").textContent = message;
  $("endpoint-status").classList.toggle("endpoint-error", error);
}
function updateEndpointActions() {
  const testing = endpointRows.some(row => row.testing);
  $("endpoints-fields").disabled = !endpointEditable || endpointBusy || benchmarkActive || benchmarkSubmitting;
  $("endpoint-benchmark-lock").hidden = !benchmarkActive && !benchmarkSubmitting;
  $("endpoint-apply").disabled = !endpointDirty || endpointBusy || testing || endpointConflict || !endpointRows.length;
  $("endpoint-reload").textContent = endpointDirty ? "Discard draft and reload" : "Reload saved";
  $("endpoint-add").disabled = testing || endpointRows.length >= 128;
  $("endpoint-reload").disabled = testing;
  for (const row of endpointRows) row.removeButton.disabled = endpointRows.length <= 1 || testing;
  updateBenchmarkRunControl();
}
function acceptEndpoints(value) {
  endpointRevision = value.revision;
  endpointEditable = value.editable;
  endpointDirty = false;
  endpointConflict = false;
  endpointRows = value.replicas.map(config => ({id: ++endpointSequence, config, testing: false, version: 0}));
  renderEndpointEditor();
  endpointStatus(value.editable ? "Saved endpoints loaded. Changes remain a draft until you apply them." : value.reason);
}
async function loadEndpoints(force = false) {
  if (endpointLoading || endpointBusy || endpointRows.some(row => row.testing)) return;
  endpointLoading = true;
  const revisionAtStart = endpointRevision;
  try {
    const value = await endpointApi("/api/endpoints");
    if (endpointBusy || (!force && revisionAtStart !== endpointRevision)) return;
    if (force || endpointRevision === null) acceptEndpoints(value);
    else if (value.revision !== endpointRevision || value.editable !== endpointEditable) {
      if (endpointDirty) {
        endpointConflict = true;
        endpointStatus("Saved configuration changed. Discard and reload your draft before applying.", true);
        updateEndpointActions();
      } else acceptEndpoints(value);
    }
  } catch (error) {
    endpointStatus(error.message, true);
  } finally {
    endpointLoading = false;
  }
}
function endpointInput(row, label, field, options = {}) {
  const wrapper = element("label", label);
  const input = element("input");
  input.id = `endpoint-${row.id}-${field}`;
  wrapper.htmlFor = input.id;
  input.type = options.type || "text";
  input.required = options.required !== false;
  input.autocomplete = "off";
  input.spellcheck = false;
  if (options.placeholder) input.placeholder = options.placeholder;
  if (options.maxLength) input.maxLength = options.maxLength;
  input.value = field === "models" ? row.config.models.join(", ") : row.config[field] || "";
  input.addEventListener("input", () => {
    row.config[field] = field === "models" ? input.value.split(",").map(value => value.trim()).filter(Boolean) : input.value.trim();
    row.version += 1;
    endpointDirty = true;
    row.feedback.textContent = "";
    if (!endpointConflict) endpointStatus("Unsaved endpoint changes.");
    updateEndpointActions();
  });
  wrapper.append(input);
  return wrapper;
}
function renderEndpointEditor() {
  $("endpoint-editor").replaceChildren();
  for (const row of endpointRows) {
    const card = element("article", undefined, "endpoint-card");
    row.feedback = element("p", "", "muted endpoint-feedback");
    row.feedback.setAttribute("role", "status");
    const fields = element("div", undefined, "endpoint-grid");
    fields.append(endpointInput(row, "Name", "name", {maxLength: 128}),
      endpointInput(row, "vLLM base URL", "url", {type: "url", placeholder: "http://127.0.0.1:8001"}),
      endpointInput(row, "Models (comma-separated)", "models", {placeholder: "Qwen/Qwen3-0.6B"}));
    const credentials = element("details", undefined, "endpoint-credentials");
    credentials.append(element("summary", "Server credentials (optional)"),
      endpointInput(row, "API key environment variable on the router", "api_key_env", {required: false, placeholder: "VLLM_API_KEY"}),
      element("p", "Enter a variable name, not the secret. The router reads the key from its environment.", "muted"));
    const actions = element("div", undefined, "endpoint-row-actions");
    const test = element("button", "Test connection", "secondary");
    test.type = "button";
    test.addEventListener("click", async () => {
      for (const input of card.querySelectorAll("input")) if (!input.reportValidity()) return;
      const version = row.version;
      row.testing = true;
      test.disabled = true;
      test.textContent = "Testing…";
      row.feedback.textContent = "Checking model availability…";
      row.feedback.classList.remove("endpoint-error");
      updateEndpointActions();
      try {
        const result = await endpointApi("/api/endpoints/test", "POST", {replica: row.config});
        if (row.version === version) {
          row.feedback.textContent = result.message + (result.models.length ? ` Available: ${result.models.join(", ")}.` : "");
          row.feedback.classList.toggle("endpoint-error", !result.ok);
        }
      } catch (error) {
        if (row.version === version) {
          row.feedback.textContent = error.message;
          row.feedback.classList.add("endpoint-error");
        }
      } finally {
        row.testing = false;
        test.disabled = false;
        test.textContent = "Test connection";
        if (row.version !== version) row.feedback.textContent = "Draft changed during the test. Test this version again.";
        updateEndpointActions();
      }
    });
    row.removeButton = element("button", "Remove", "secondary danger");
    row.removeButton.type = "button";
    row.removeButton.addEventListener("click", () => {
      endpointRows = endpointRows.filter(item => item !== row);
      endpointDirty = true;
      renderEndpointEditor();
      if (!endpointConflict) endpointStatus("Unsaved endpoint removal. Apply to stop sending new requests to this endpoint.");
    });
    actions.append(test, row.removeButton);
    card.append(fields, credentials, actions, row.feedback);
    $("endpoint-editor").append(card);
  }
  updateEndpointActions();
}
$("endpoint-add").addEventListener("click", () => {
  let suffix = endpointRows.length + 1;
  while (endpointRows.some(row => row.config.name === `replica-${suffix}`)) suffix += 1;
  endpointRows.push({id: ++endpointSequence, config: {name: `replica-${suffix}`, url: "",
    models: [...(endpointRows[0]?.config.models || ["Qwen/Qwen3-0.6B"])], capacity: 1}, testing: false, version: 0});
  endpointDirty = true;
  renderEndpointEditor();
  if (!endpointConflict) endpointStatus("New endpoint is a draft. Enter its deployed server URL and model names.");
  $("endpoint-editor").lastElementChild.querySelector('input[type="url"]').focus();
});
$("endpoint-reload").addEventListener("click", () => loadEndpoints(true));
$("endpoints-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (!endpointDirty || endpointBusy || endpointConflict || benchmarkActive || benchmarkSubmitting || endpointRows.some(row => row.testing)) return;
  endpointBusy = true;
  updateEndpointActions();
  endpointStatus("Saving and applying endpoints…");
  try {
    const result = await endpointApi("/api/endpoints", "PUT", {
      revision: endpointRevision, replicas: endpointRows.map(row => row.config)
    });
    acceptEndpoints(result);
    endpointStatus("Endpoints saved and applied. Live health checks will refresh automatically.");
    refresh();
  } catch (error) {
    if (error.status === 409) endpointConflict = true;
    endpointStatus(error.message, true);
  } finally {
    endpointBusy = false;
    updateEndpointActions();
  }
});
function updateBenchmarkRunControl() {
  const traceMode = $("benchmark-mode").value === "fixed_arrival";
  const trace = benchmarkOptions?.traces.find(item => item.id === $("benchmark-trace").value);
  const ready = benchmarkOptions?.can_run && (traceMode ? Boolean(trace) : Boolean($("benchmark-model").value));
  $("benchmark-run").disabled = !ready || benchmarkActive || benchmarkSubmitting || endpointDirty || endpointBusy;
  $("benchmark-draft-note").hidden = !endpointDirty && !endpointBusy;
}
function updateBenchmarkForm() {
  const traceMode = $("benchmark-mode").value === "fixed_arrival";
  for (const [field, input] of [["model", "model"], ["workers", "workers"], ["duration", "duration"], ["seed", "seed"]]) {
    $(`benchmark-${field}-field`).hidden = traceMode;
    $(`benchmark-${input}`).disabled = traceMode;
  }
  for (const name of ["trace", "inflight"]) {
    $(`benchmark-${name}-field`).hidden = !traceMode;
    $(`benchmark-${name}`).disabled = !traceMode;
  }
  const hardware = $("benchmark-environment").value === "hardware";
  $("benchmark-gpu").disabled = !hardware;
  if (!hardware) $("benchmark-gpu").checked = false;
  const trace = benchmarkOptions?.traces.find(item => item.id === $("benchmark-trace").value);
  $("benchmark-trace-info").hidden = !traceMode;
  $("benchmark-trace-info").textContent = trace ?
    `${trace.requests} requests · ${trace.arrival_seconds.toFixed(2)} s of arrivals · models: ${trace.models.join(", ")} · SHA-256: ${trace.sha256}` :
    "Choose a trace saved under the results/workloads directory, then reload inputs.";
  const target = traceMode ? (trace ? `${trace.requests} stored requests over ${trace.arrival_seconds.toFixed(2)} s, then drain` : "Choose a saved trace") :
    `${$("benchmark-workers").value} workers for ${$("benchmark-duration").value} s · ${$("benchmark-model").value || "choose a model"}`;
  $("benchmark-preview").textContent = `${$("benchmark-policy").value} · ${target} · ${$("benchmark-warmup").value} warm-up requests per model, excluded from measurement.`;
  $("benchmark-fields").disabled = !benchmarkOptions?.can_run || benchmarkActive || benchmarkSubmitting;
  $("benchmark-run").textContent = benchmarkSubmitting ? "Starting…" : benchmarkActive ? "Benchmark running…" : "Run benchmark";
  $("benchmark-inputs-reload").disabled = benchmarkActive || benchmarkSubmitting || benchmarkOptionsLoading;
  updateEndpointActions();
}
async function loadBenchmarkOptions(force = false) {
  if (benchmarkOptionsLoading || (benchmarkOptions && !force) || benchmarkActive || benchmarkSubmitting) return;
  benchmarkOptionsLoading = true;
  $("benchmark-inputs-reload").disabled = true;
  try {
    const value = await endpointApi("/api/benchmark/options");
    const first = benchmarkOptions === null;
    benchmarkOptions = value;
    const oldModel = $("benchmark-model").value;
    $("benchmark-model").replaceChildren(...value.models.map(model => {
      const option = element("option", model); option.value = model; return option;
    }));
    if (value.models.includes(oldModel)) $("benchmark-model").value = oldModel;
    const oldTrace = $("benchmark-trace").value;
    const placeholder = element("option", "Choose a saved trace"); placeholder.value = "";
    $("benchmark-trace").replaceChildren(placeholder, ...value.traces.map(trace => {
      const option = element("option", `${trace.id} · ${trace.requests} requests`); option.value = trace.id; return option;
    }));
    if (value.traces.some(trace => trace.id === oldTrace)) $("benchmark-trace").value = oldTrace;
    else if (first && value.traces.length) $("benchmark-trace").value = value.traces[0].id;
    if (first) $("benchmark-policy").value = value.defaults.policy;
    $("benchmark-options-status").textContent = value.reason ||
      (value.trace_errors.length ? `${value.trace_errors.length} workload file(s) could not be loaded. Check those files before using them.` :
        "Ready. One dashboard benchmark can run at a time; closing this page does not stop it.");
    $("benchmark-options-status").classList.toggle("endpoint-error", !value.can_run || value.trace_errors.length > 0);
    refreshBenchmarkStatus();
  } catch (error) {
    $("benchmark-options-status").textContent = error.message;
    $("benchmark-options-status").classList.add("endpoint-error");
  } finally {
    benchmarkOptionsLoading = false;
    updateBenchmarkForm();
  }
}
function showBenchmarkStatus(value) {
  const wasActive = benchmarkActive;
  benchmarkActive = value.active;
  const job = value.job;
  $("benchmark-job-status").textContent = job ? ({queued: "Queued", running: "Running", completed: "Completed", failed: "Failed"}[job.status] || job.status) : "Idle";
  $("benchmark-job-status").classList.toggle("endpoint-error", job?.status === "failed");
  $("benchmark-job-message").textContent = job ? `${job.message}${job.elapsed_seconds == null ? "" : ` · ${job.elapsed_seconds.toFixed(1)} s elapsed`}` : "No dashboard benchmark has started.";
  $("benchmark-job-output").textContent = job?.directory ? `Saved: ${value.results_directory}/${job.directory} · run ${job.run_id}` : "";
  const signature = job ? `${job.job_id}:${job.status}` : "";
  const changed = signature !== benchmarkLastSignature;
  benchmarkLastSignature = signature;
  updateBenchmarkForm();
  if (changed && job && !value.active) refresh();
  if (wasActive && !value.active) { loadEndpoints(); loadBenchmarkOptions(true); }
}
async function refreshBenchmarkStatus() {
  if (!benchmarkOptions || benchmarkStatusLoading || benchmarkSubmitting) return;
  benchmarkStatusLoading = true;
  const version = benchmarkStatusVersion;
  try {
    const value = await endpointApi("/api/benchmark");
    if (version === benchmarkStatusVersion && !benchmarkSubmitting) showBenchmarkStatus(value);
  } catch (error) {
    $("benchmark-job-message").textContent = error.message;
  } finally { benchmarkStatusLoading = false; }
}
$("benchmark-inputs-reload").addEventListener("click", () => loadBenchmarkOptions(true));
$("benchmark-form").addEventListener("input", updateBenchmarkForm);
$("benchmark-form").addEventListener("change", updateBenchmarkForm);
$("benchmark-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (!benchmarkOptions?.can_run || benchmarkActive || benchmarkSubmitting || endpointDirty || endpointBusy) return;
  const body = {mode: $("benchmark-mode").value, policy: $("benchmark-policy").value,
    environment: $("benchmark-environment").value, gpu: $("benchmark-gpu").checked,
    warmup: Number($("benchmark-warmup").value), model_revision: $("benchmark-model-revision").value,
    image_digest: $("benchmark-image-digest").value};
  if (body.mode === "closed_loop") Object.assign(body, {model: $("benchmark-model").value,
    concurrency: Number($("benchmark-workers").value), duration: Number($("benchmark-duration").value), seed: Number($("benchmark-seed").value)});
  else {
    const trace = benchmarkOptions.traces.find(item => item.id === $("benchmark-trace").value);
    if (!trace) return;
    Object.assign(body, {trace_id: trace.id, trace_sha256: trace.sha256, max_inflight: Number($("benchmark-inflight").value)});
  }
  benchmarkSubmitting = true;
  benchmarkStatusVersion += 1;
  updateBenchmarkForm();
  try {
    const value = await endpointApi("/api/benchmark", "POST", body);
    showBenchmarkStatus(value);
    $("benchmark-options-status").textContent = "The run uses the selected policy temporarily. Endpoint changes are locked until it finishes.";
  } catch (error) {
    $("benchmark-options-status").textContent = error.message;
    $("benchmark-options-status").classList.add("endpoint-error");
    if (error.status === 409) refreshBenchmarkStatus();
  } finally {
    benchmarkSubmitting = false;
    updateBenchmarkForm();
    refreshBenchmarkStatus();
  }
});
function baseFilteredRuns() {
  return runs.filter(run => ($("environment").value === "all" || run.environment === $("environment").value) &&
    ($("scenario").value === "all" || (run.scenario || run.mode) === $("scenario").value));
}
function filteredRuns() {
  const key = $("comparison").value;
  return baseFilteredRuns().filter(run => key === "individual" || comparisonKey(run.context) === key);
}
function comparisonKey(context = {}, mode = $("comparison-mode").value) {
  return context[mode === "exploratory" ? "exploratory_comparison_key" : "comparison_key"];
}
function visibleGroups(mode = $("comparison-mode").value) {
  const keys = new Set(baseFilteredRuns().map(run => comparisonKey(run.context, mode)).filter(Boolean));
  const latestRun = group => group.run_ids?.slice(-1)[0] || "";
  return comparisonCatalog[mode].filter(group => keys.has(group.id))
    .sort((a, b) => latestRun(b).localeCompare(latestRun(a)));
}
function ms(value) { return Number.isFinite(value) ? (value * 1000).toFixed(1) + " ms" : "Unknown"; }
function count(value) { return value == null ? "Unknown" : String(value); }
function energyValue(value, unit, digits = 2) { return Number.isFinite(value) ? `${value.toFixed(digits)} ${unit}` : "Unavailable"; }
const POLICY_COLORS = {round_robin: "#efc37c", capacity: "#a4c8ee", kubeai: "#9cddbd", direct: "#a2b1bd"};
const policyName = policy => ({round_robin: "Round robin", capacity: "Capacity", kubeai: "KubeAI", direct: "Direct"}[policy] || policy);
const runEnergy = (run, key) => run.context?.gpu?.energy?.status === "estimated" ? run.context.gpu.energy[key] : null;
const COMPARISON_METRICS = [
  {id: "p95", title: "P95 latency", unit: "ms", scale: 1000, digits: 1, mean: "latency_p95_mean_seconds",
    read: run => run.latency_p95_seconds, description: "Mean of per-run P95 values. Lower values mean faster responses.",
    missing: "Completed responses with recorded latency are required."},
  {id: "ttft", title: "TTFT P95", unit: "ms", scale: 1000, digits: 1, mean: "ttft_p95_mean_seconds",
    read: run => run.ttft_p95_seconds, description: "Time to the first generated token. Lower values mean less waiting.",
    missing: "Streaming responses with recorded first-token timing are required."},
  {id: "throughput", title: "Throughput", unit: "req/s", scale: 1, digits: 2, mean: "completed_rps_mean",
    read: run => run.completed_rps, description: "Completed requests per second. Read alongside failures and drops.",
    missing: "Recorded completions and replay duration are required."},
  {id: "power", title: "GPU power", unit: "W", scale: 1, digits: 2, mean: "power_mean_watts_mean",
    read: run => runEnergy(run, "power_mean_watts"), description: "Mean gross GPU board power. Interpret with the amount of completed work.",
    missing: "Complete GPU power coverage is required for every run."},
  {id: "energy-request", title: "Energy per request", unit: "J/request", scale: 1, digits: 2, mean: "joules_per_completed_request_mean",
    read: run => runEnergy(run, "joules_per_completed_request"), description: "Gross GPU energy per completed request. Lower values use less energy per response.",
    missing: "Complete power coverage and positive completed-request counts are required."},
  {id: "energy-tokens", title: "Energy per 1000 tokens", unit: "J/1000 tokens", scale: 1, digits: 2, mean: "joules_per_1000_output_tokens_mean",
    read: run => runEnergy(run, "joules_per_1000_output_tokens"), description: "Gross GPU energy per 1000 output tokens. This does not measure answer quality.",
    missing: "Complete power coverage and complete positive output-token usage are required."}
];
let renderedChartSignature = "";
function svgElement(tag, attributes = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}
function scaledMetric(value, metric) {
  const scaled = Number.isFinite(value) && value >= 0 ? value * metric.scale : null;
  return Number.isFinite(scaled) ? scaled : null;
}
function metricCard(group, metric, groupRuns) {
  const card = element("article", undefined, "metric-card");
  const title = element("h3", metric.title); title.id = `metric-${metric.id}-title`;
  title.append(element("span", metric.unit, "metric-unit"));
  card.setAttribute("aria-labelledby", title.id);
  card.append(title, element("p", metric.description, "metric-description"));
  const rows = group.policies.map(policy => {
    const measured = groupRuns.filter(run => run.policy === policy.policy);
    const points = measured.map(run => ({run, value: scaledMetric(metric.read(run), metric)})).filter(point => point.value !== null);
    const complete = measured.length === policy.runs && points.length === policy.runs;
    return {policy, points, mean: complete ? scaledMetric(policy[metric.mean], metric) : null};
  });
  const values = rows.flatMap(row => [...row.points.map(point => point.value), ...(row.mean === null ? [] : [row.mean])]);
  const readout = element("p", `Hover or focus a dot for its run and value; select it to inspect. ${metric.missing}`, "metric-readout");
  readout.setAttribute("aria-live", "polite");
  if (!values.length) {
    const missing = element("ul", undefined, "metric-missing");
    for (const row of rows) {
      const item = element("li");
      item.append(element("span", policyName(row.policy.policy)), element("span", `Unavailable · 0/${row.policy.runs} values`));
      missing.append(item);
    }
    card.append(missing, element("p", metric.missing, "metric-readout"));
    return card;
  }
  const maximum = Math.max(...values);
  const magnitude = maximum > 0 ? 10 ** Math.floor(Math.log10(maximum)) : 1;
  const rounded = maximum > 0 ? Math.ceil(maximum / magnitude * 2) / 2 * magnitude : 1;
  const domain = Number.isFinite(rounded) && rounded > 0 ? rounded : maximum || 1;
  const left = 130, width = 325, bottom = 10 + rows.length * 56;
  const x = value => left + value / domain * width;
  const svg = svgElement("svg", {viewBox: `0 0 640 ${bottom + 30}`, role: "group",
    "aria-label": `${group.kind === "exploratory" ? "Exploratory comparison. " : ""}${metric.title} in ${metric.unit}. Bars show policy means; dots show individual runs.`});
  for (let tick = 0; tick <= 4; tick++) {
    const value = domain * (tick / 4);
    svg.append(svgElement("line", {x1: x(value), x2: x(value), y1: 0, y2: bottom, stroke: "#2b3a49"}),
      svgElement("text", {x: x(value), y: bottom + 19, "text-anchor": tick === 0 ? "start" : tick === 4 ? "end" : "middle", class: "metric-tick"},
        value !== 0 && Math.abs(value) < .01 ? value.toExponential(1) : Number(value.toFixed(2)).toLocaleString("en")));
  }
  rows.forEach((row, index) => {
    const top = 10 + index * 56;
    const color = POLICY_COLORS[row.policy.policy] || "#a2b1bd";
    svg.append(svgElement("text", {x: 0, y: top + 19, class: "metric-policy"}, policyName(row.policy.policy)),
      svgElement("text", {x: 0, y: top + 39, class: "metric-count"}, `${row.policy.runs} run(s) · ${row.points.length}/${row.policy.runs} values`));
    if (row.mean !== null) {
      const bar = svgElement("rect", {x: left, y: top + 6, width: row.mean / domain * width, height: 16, rx: 3, fill: color, opacity: .8});
      bar.append(svgElement("title", {}, `${policyName(row.policy.policy)} mean: ${row.mean.toFixed(metric.digits)} ${metric.unit}, across ${row.policy.runs} run(s).`));
      svg.append(bar);
    }
    svg.append(svgElement("text", {x: 628, y: top + 19, "text-anchor": "end", class: row.mean === null ? "metric-unavailable" : "metric-value"},
      row.mean === null ? "Unavailable" : `${row.mean.toFixed(metric.digits)} ${metric.unit}`));
    row.points.forEach((point, pointIndex) => {
      // Vertical stacking separates coincident repeat values; the horizontal axis carries the metric.
      const lanes = Math.min(5, row.points.length);
      const offset = (pointIndex % lanes - (lanes - 1) / 2) * 7;
      const label = `${policyName(row.policy.policy)} · ${point.run.run_id} · ${point.value.toFixed(metric.digits)} ${metric.unit}`;
      const dot = svgElement("circle", {cx: x(point.value), cy: top + 39 + offset, r: 2.5,
        fill: "#17222e", stroke: color, "stroke-width": 1.5, tabindex: 0, role: "button", class: "metric-point", "aria-label": label + ". Inspect run."});
      dot.append(svgElement("title", {}, label));
      dot.addEventListener("pointerenter", () => { readout.textContent = label; });
      dot.addEventListener("focus", () => { readout.textContent = label; });
      dot.addEventListener("click", () => inspectRun(point.run));
      dot.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); inspectRun(point.run); }
      });
      svg.append(dot);
    });
  });
  const plot = element("div", undefined, "metric-plot");
  plot.append(svg);
  card.append(plot, readout);
  return card;
}
function setupLabel(context = {}) {
  const load = context.mode === "closed_loop" ? `${count(context.workers)} worker(s) · ${count(context.duration_seconds)} s` :
    context.mode === "fixed_arrival" ? `Trace ${(context.trace_sha256 || "unknown").slice(0, 8)} · ${count(context.input_rows)} requests` : "Unknown workload";
  return `${load} · ${count(context.endpoint_count)} endpoint(s)`;
}
function refreshComparisonPicker() {
  const previous = $("comparison").value;
  const visible = visibleGroups();
  const individual = element("option", "Individual runs — no averaging"); individual.value = "individual";
  $("comparison").replaceChildren(individual);
  for (const group of visible) {
    const sampling = {recorded: "GPU sampled", disabled: "GPU sampling off"}[group.context.gpu?.status] || "GPU sampling unknown";
    const prefix = group.kind === "exploratory" ? "Exploratory · " : "";
    const option = element("option", `${prefix}${group.environment} · ${setupLabel(group.context)} · ${sampling} · ${group.policies.length} policy/policies · ${group.id.slice(0, 8)}`);
    option.value = group.id;
    $("comparison").append(option);
  }
  if (pickLatestComparison && visible.length) $("comparison").value = visible[0].id;
  else if ([...$("comparison").options].some(option => option.value === previous)) $("comparison").value = previous;
  if (comparisonsReady) pickLatestComparison = false;
  const eligible = baseFilteredRuns().filter(run => run.context?.exploratory_comparison_key);
  $("exploratory-availability").hidden = !eligible.length || $("comparison-mode").value === "exploratory";
  $("exploratory-availability-text").textContent = `${eligible.length} saved run(s) have matching workload context but missing declared model/image provenance. They are available in Exploratory comparison; a selected Recorded provenance setup keeps its own results. ${comparisonIssues.exploratory}`;
  $("show-exploratory").disabled = !visibleGroups("exploratory").length;
}
async function inspectRun(run) {
  const version = ++detailVersion;
  const context = run.context || {};
  $("detail").hidden = false;
  $("detail-title").textContent = run.run_id;
  const messages = [...(context.missing?.length ? [`Missing: ${context.missing.join(", ")}. ${context.exploratory_comparison_key ? "Exploratory comparison is available; deployment provenance remains incomplete." : "Displayed individually."}`] : []), ...(context.warnings || [])];
  $("detail-note").textContent = messages.join(" ") || "Recorded settings are available. A repeatable protocol is still needed before claiming an advantage.";
  $("detail-context").replaceChildren();
  const values = [["Mode / load", setupLabel(context)], ["Models in saved inputs", context.models?.join(", ") || "Unknown"],
    ["Warm-up per model", count(context.warmup_per_model)], ["Input hash", context.trace_sha256 || "Unknown"],
    ["Source hash", context.source_sha256 || "Unknown"], ["Declared model revision", context.model_revision || "Unknown"],
    ["Declared image", context.image_digest || "Unknown"], ["Streaming", context.stream == null ? "Unknown" : context.stream ? "Enabled" : "Disabled"],
    ["Client in-flight limit", count(context.client_max_inflight)], ["Local GPU samples", context.gpu?.status || "Unknown"]];
  if (context.routing?.kind === "direct") values.push(["Direct endpoint", context.routing.url]);
  for (const replica of context.routing?.configuration?.replicas || []) {
    values.push([`Endpoint ${replica.name}`, `${replica.url || "Temporary emulator URL"} · capacity ${replica.capacity} · ${(replica.models || []).join(", ")}`]);
  }
  for (const [label, value] of values) $("detail-context").append(element("dt", label), element("dd", value));
  const energy = context.gpu?.energy || {};
  $("detail-energy-status").textContent = `${energy.status || "unavailable"}: ${energy.message || "No saved GPU power measurements."}`;
  $("detail-energy").replaceChildren();
  for (const [label, value] of [["Mean power", energyValue(energy.power_mean_watts, "W")],
    ["Sampled max power", energyValue(energy.power_max_sampled_watts, "W")], ["Estimated energy", energyValue(energy.energy_joules, "J")],
    ["Per completed request", energyValue(energy.joules_per_completed_request, "J/request")],
    ["Per 1000 output tokens", energyValue(energy.joules_per_1000_output_tokens, "J/1000 tokens")],
    ["Power coverage", `${energyValue(energy.coverage_percent, "%", 1)} · ${energyValue(energy.covered_seconds, "s")} of ${energyValue(energy.duration_seconds, "s")}`],
    ["Covered energy only", energyValue(energy.covered_energy_joules, "J")],
    ["Scope", energy.scope || "All physical GPUs on the benchmark host"],
    ["Integration", `${energy.method || "Unavailable"} · maximum gap ${energyValue(energy.max_gap_seconds, "s")}`]]) {
    $("detail-energy").append(element("dt", label), element("dd", value));
  }
  $("detail-gpu").replaceChildren();
  for (const gpu of context.gpu?.devices || []) {
    const card = element("article", undefined, "replica");
    card.append(element("h4", gpu.name), element("p", gpu.uuid, "model"),
      element("p", `Benchmark host · ${gpu.samples} samples · driver ${gpu.driver_version}`, "muted"),
      element("p", `VRAM used: ${gpu.memory_used_min_mib}–${gpu.memory_used_max_mib} MiB of ${gpu.memory_total_mib} MiB`, "muted"),
      element("p", `Utilization: mean ${gpu.utilization_mean_percent.toFixed(1)}%, sampled max ${gpu.utilization_max_percent.toFixed(1)}%`, "muted"),
      element("p", "This is one physical device record, not a separate allocation for each endpoint.", "muted"));
    const deviceEnergy = energy.devices?.find(device => device.uuid === gpu.uuid);
    if (deviceEnergy) {
      card.append(element("p", `Mean power: ${energyValue(deviceEnergy.power_mean_watts, "W")} · sampled max: ${energyValue(deviceEnergy.power_max_sampled_watts, "W")}`, "muted"),
        element("p", `Estimated energy: ${energyValue(deviceEnergy.energy_joules, "J")} · coverage: ${energyValue(deviceEnergy.coverage_percent, "%", 1)}`, "muted"));
    }
    $("detail-gpu").append(card);
  }
  $("detail-body").textContent = JSON.stringify(run, null, 2);
  $("detail-replicas").replaceChildren();
  $("detail-replicas-status").textContent = "Loading saved request measurements…";
  $("detail-replicas-status").classList.remove("endpoint-error");
  $("detail").scrollIntoView({behavior: "smooth", block: "nearest"});
  try {
    const detail = await endpointApi("/api/runs/" + encodeURIComponent(run.run_id));
    if (version !== detailVersion) return;
    const breakdown = detail.replica_breakdown;
    $("detail-replicas-status").textContent = breakdown.message;
    $("detail-replicas-status").classList.toggle("endpoint-error", breakdown.status !== "recorded");
    for (const replica of breakdown.replicas) {
      const row = element("tr");
      row.append(element("td", replica.replica ?? "No replica recorded"));
      const requests = element("td");
      requests.append(element("span", `${count(replica.requests)} · ${Number.isFinite(replica.request_share_percent) ? replica.request_share_percent.toFixed(1) + "%" : "Unknown share"}`));
      if (Number.isFinite(replica.request_share_percent)) {
        const progress = element("progress", undefined, "request-share");
        progress.max = 100; progress.value = replica.request_share_percent;
        progress.setAttribute("aria-label", `Recorded request share: ${replica.request_share_percent.toFixed(1)} percent`);
        requests.append(progress);
      }
      row.append(requests);
      for (const value of [count(replica.completed), ms(replica.latency_p50_seconds), ms(replica.latency_p95_seconds),
        ms(replica.ttft_p95_seconds), `${count(replica.latency_samples)}/${count(replica.completed)}`,
        `${count(replica.ttft_samples)}/${count(replica.completed)}`, count(replica.errors), count(replica.dropped), count(replica.truncated)]) row.append(element("td", value));
      $("detail-replicas").append(row);
    }
  } catch (error) {
    if (version !== detailVersion) return;
    $("detail-replicas-status").textContent = error.message;
    $("detail-replicas-status").classList.add("endpoint-error");
  }
}
function renderRuns() {
  const mode = $("comparison-mode").value;
  comparisonGroups = comparisonCatalog[mode];
  comparisonIssue = comparisonIssues[mode];
  refreshComparisonPicker();
  const selected = filteredRuns();
  $("runs").replaceChildren();
  $("empty").hidden = selected.length > 0;
  if (!selected.length) $("empty").textContent = runs.length ? "No runs match these filters." : "No runs yet. Run python3 -m kubeai demo from the project directory.";
  for (const run of [...selected].reverse()) {
    const row = element("tr");
    const name = element("td", `${run.scenario || run.mode} · ${run.run_id}`);
    name.title = run.run_id;
    row.append(name, element("td", run.policy));
    const source = element("td");
    source.append(element("span", run.environment, `pill ${run.environment === "emulator" ? "emulator" : ""}`));
    const setup = element("td", undefined, "run-setup");
    const context = run.context || {};
    setup.append(element("span", setupLabel(context)), element("small", context.exploratory_comparison_key ? "Exploratory comparison available" :
      ({matched_settings: "Recorded context complete", individual_only: "Individual only", excluded: "Excluded from comparisons"}[context.status] || "Unknown context")));
    setup.title = (context.missing || []).join(", ");
    row.append(source, setup, element("td", `${count(run.completed)}/${count(run.requests)}`), element("td", ms(run.latency_p50_seconds)),
      element("td", ms(run.latency_p95_seconds)), element("td", ms(run.ttft_p95_seconds)),
      element("td", Number.isFinite(run.completed_rps) ? display(run.completed_rps) : "Unknown"),
      element("td", count(run.errors)), element("td", count(run.dropped)), element("td", count(run.truncated)));
    const energy = context.gpu?.energy || {};
    row.append(element("td", energyValue(energy.joules_per_completed_request, "J")), element("td", energyValue(energy.coverage_percent, "%", 1)));
    const action = element("td");
    const button = element("button", "Inspect ↗");
    button.type = "button";
    button.addEventListener("click", () => inspectRun(run));
    action.append(button);
    row.append(action);
    $("runs").append(row);
  }
  chart(selected);
}
function chart(selected) {
  const group = comparisonGroups.find(item => item.id === $("comparison").value);
  const exploratory = group?.kind === "exploratory";
  $("comparison-warning").hidden = !exploratory;
  $("comparison-warning").textContent = exploratory ? `Exploratory comparison — missing declared ${group.missing_provenance.join(" and ")}. Matching recorded settings does not establish that the deployed revisions/images were identical. Treat these measurements as exploratory; no validated ranking or score is inferred.` : "";
  $("comparison-table-wrapper").hidden = !group;
  $("comparison-energy-wrapper").hidden = !group;
  $("comparison-details").hidden = !group;
  $("chart-legend").hidden = !group;
  $("comparison-outcomes").hidden = !group;
  $("comparison-rows").replaceChildren();
  $("comparison-energy-rows").replaceChildren();
  let data;
  if (group) {
    $("comparison-note").textContent = `${exploratory ? "Exploratory comparison. " : ""}${setupLabel(group.context)}. Means summarize per-run percentiles, not pooled requests. Repeats per policy: ${group.policies.map(item => `${item.policy}=${item.runs}`).join(", ")}. Small samples remain exploratory.`;
    const total = key => group.policies.length && group.policies.every(policy => Number.isFinite(policy[key])) ?
      group.policies.reduce((sum, policy) => sum + policy[key], 0) : null;
    $("comparison-outcomes").textContent = `${count(total("runs"))} runs · ${count(total("completed"))}/${count(total("requests"))} responses · ${count(total("errors"))} failures · ${count(total("dropped"))} drops · ${count(total("truncated"))} token-limited responses`;
    for (const policy of group.policies) {
      const row = element("tr");
      const range = Number.isFinite(policy.latency_p95_min_seconds) && Number.isFinite(policy.latency_p95_max_seconds) ?
        `${ms(policy.latency_p95_min_seconds)}–${ms(policy.latency_p95_max_seconds)}` : "Unknown";
      for (const text of [policy.policy, policy.runs, ms(policy.latency_p50_mean_seconds), ms(policy.latency_p95_mean_seconds), range,
        ms(policy.ttft_p95_mean_seconds), Number.isFinite(policy.completed_rps_mean) ? display(policy.completed_rps_mean) : "Unknown",
        `${count(policy.completed)}/${count(policy.requests)}`, count(policy.errors), count(policy.dropped), count(policy.truncated)]) row.append(element("td", text));
      $("comparison-rows").append(row);
      const energyRow = element("tr");
      for (const value of [policy.policy, `${count(policy.energy_runs)}/${policy.runs}`,
        energyValue(policy.power_mean_watts_mean, "W"), energyValue(policy.energy_joules_mean, "J"),
        energyValue(policy.joules_per_completed_request_mean, "J/request"), energyValue(policy.joules_per_1000_output_tokens_mean, "J/1000 tokens")]) {
        energyRow.append(element("td", value));
      }
      $("comparison-energy-rows").append(energyRow);
    }
  } else {
    $("comparison-note").textContent = comparisonIssue || "Each bar is one run; no values are averaged. Choose an experiment setup to compare policies. Hardware runs missing only deployment provenance are available in Exploratory comparison; other missing context keeps runs individual.";
    data = selected.filter(run => Number.isFinite(run.latency_p95_seconds) && run.latency_p95_seconds >= 0 && !run.interrupted && run.valid_for_comparison !== false && run.context?.status !== "excluded")
      .map(run => ({policy: run.policy, label: `${run.policy} · ${run.context?.workers ? run.context.workers + "w" : "trace"} · ${run.run_id.slice(-8)}`,
        mean: run.latency_p95_seconds, min: null, max: null, title: `${run.run_id}. ${setupLabel(run.context)}. Individual measurement.`}));
  }
  const identifiers = new Set(group?.run_ids || []);
  const chartRuns = group ? selected.filter(run => identifiers.has(run.run_id)) : selected;
  const signature = JSON.stringify({mode: $("comparison-mode").value, group: group?.id, policies: group?.policies,
    runs: chartRuns.map(run => [run.run_id, run.policy, setupLabel(run.context), run.interrupted, run.valid_for_comparison, run.context?.status,
      ...COMPARISON_METRICS.map(metric => metric.read(run))])});
  // Keep focused points and their readouts when an automatic refresh brings no new data.
  if (signature === renderedChartSignature) return;
  $("chart").replaceChildren();
  if (group) {
    for (const metric of COMPARISON_METRICS) $("chart").append(metricCard(group, metric, chartRuns));
    renderedChartSignature = signature;
    return;
  }
  const individual = element("div", undefined, "chart individual-chart");
  $("chart").append(individual);
  if (!data.length) {
    individual.append(element("p", "No recorded P95 values for this selection. Failed requests and missing measurements remain visible in the tables.", "muted"));
    renderedChartSignature = signature;
    return;
  }
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 1000 ${data.length * 40 + 30}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Individual run P95 latency, without averaging");
  const max = Math.max(...data.map(item => item.max ?? item.mean), .001);
  data.forEach((item, index) => {
    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", "0"); label.setAttribute("y", String(index * 40 + 24));
    label.textContent = item.label;
    const bar = document.createElementNS(ns, "rect");
    bar.setAttribute("x", "330"); bar.setAttribute("y", String(index * 40 + 8));
    bar.setAttribute("width", String(Math.max(1, item.mean / max * 570))); bar.setAttribute("height", "24");
    bar.setAttribute("rx", "3"); bar.setAttribute("fill", POLICY_COLORS[item.policy] || "#a2b1bd");
    const title = document.createElementNS(ns, "title");
    title.textContent = item.title;
    bar.append(title);
    const value = document.createElementNS(ns, "text");
    value.setAttribute("x", "920"); value.setAttribute("y", String(index * 40 + 24));
    value.textContent = ms(item.mean);
    svg.append(label, bar, value);
    if (Number.isFinite(item.min) && Number.isFinite(item.max) && item.min !== item.max) {
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", String(330 + item.min / max * 570)); line.setAttribute("x2", String(330 + item.max / max * 570));
      line.setAttribute("y1", String(index * 40 + 20)); line.setAttribute("y2", String(index * 40 + 20));
      line.setAttribute("stroke", "#edf2f3"); line.setAttribute("stroke-width", "3"); svg.append(line);
    }
  });
  individual.append(svg);
  renderedChartSignature = signature;
}
async function refresh() {
  if (loading) return;
  loading = true;
  try {
    const headers = apiHeaders();
    const get = async path => {
      const response = await fetch(path, {headers, cache: "no-store"});
      if (!response.ok) throw new Error(response.status === 401 ? "Enter the configured router bearer token to view measurements." : `Router returned HTTP ${response.status}`);
      return response.json();
    };
    const getComparisons = async mode => {
      try {
        const value = await get(mode === "recorded" ? "/api/comparisons" : "/api/comparisons?mode=exploratory");
        if (mode === "exploratory" && value.mode !== mode) throw new Error("Restart the router to enable Exploratory comparison");
        if (!Array.isArray(value.groups)) throw new Error("Unreadable comparison data");
        return {groups: value.groups.filter(group => mode === "exploratory" ? group.kind === mode : group.kind !== "exploratory"), issue: ""};
      } catch (error) {
        return {groups: [], issue: `${mode === "exploratory" ? "Exploratory" : "Recorded provenance"} comparison unavailable: ${error.message}. Individual runs remain visible.`};
      }
    };
    const [state, history, recorded, exploratory] = await Promise.all([get("/api/state"), get("/api/runs"),
      getComparisons("recorded"), getComparisons("exploratory")]);
    runs = history.runs;
    comparisonCatalog = {recorded: recorded.groups, exploratory: exploratory.groups};
    comparisonIssues = {recorded: recorded.issue, exploratory: exploratory.issue};
    comparisonsReady = true;
    $("error").hidden = !state.control_error;
    $("error").textContent = state.control_error ? "Configuration/discovery: " + state.control_error : "";
    $("connection").textContent = "● Router connected";
    $("updated").textContent = "Updated " + new Date().toLocaleTimeString() + " · refreshes every 10 seconds";
    $("policy").textContent = state.policy;
    $("healthy").textContent = `${state.replicas.filter(r => r.healthy).length} / ${state.replicas.length}`;
    $("inflight").textContent = state.replicas.reduce((sum, r) => sum + r.inflight, 0);
    $("run-count").textContent = runs.length;
    $("replicas").replaceChildren();
    if (!state.replicas.length) $("replicas").append(element("p", "No discovered endpoints. Check service labels and discovery status.", "muted"));
    for (const replica of state.replicas) {
      const card = element("article", undefined, "replica");
      const heading = element("h3");
      heading.append(element("span", undefined, `dot ${replica.healthy ? "up" : ""}`), document.createTextNode(replica.name));
      card.append(heading, element("p", replica.models.join(", ") || "No verified model", "model"));
      if (replica.url) card.append(element("p", replica.url, "replica-url"));
      const details = element("dl");
      for (const [key, value] of [["Health", replica.healthy ? "Healthy" : "Unavailable"], ["In flight", replica.inflight], ["Relative capacity", replica.capacity], ["Telemetry", replica.metrics_fresh ? "Fresh" : "Missing / stale"], ["Observed queue", replica.metrics_fresh ? (replica.metrics.waiting ?? "—") : "—"]]) {
        details.append(element("dt", key), element("dd", String(value)));
      }
      card.append(details); $("replicas").append(card);
    }
    const priorScenario = $("scenario").value;
    $("scenario").replaceChildren();
    const all = element("option", "All scenarios"); all.value = "all"; $("scenario").append(all);
    for (const name of [...new Set(runs.map(r => r.scenario || r.mode))].sort()) {
      const option = element("option", name); option.value = name; $("scenario").append(option);
    }
    if ([...$("scenario").options].some(o => o.value === priorScenario)) $("scenario").value = priorScenario;
    renderRuns();
    loadEndpoints();
    const modelSignature = JSON.stringify([state.control_error, Boolean(state.configuration.discovery),
      [...new Set(state.replicas.filter(replica => replica.healthy).flatMap(replica => replica.models))].sort()]);
    const modelsChanged = modelSignature !== benchmarkModelsSignature;
    benchmarkModelsSignature = modelSignature;
    loadBenchmarkOptions(modelsChanged);
    refreshBenchmarkStatus();
  } catch (error) {
    $("connection").textContent = "● Measurements unavailable";
    $("error").textContent = error.message; $("error").hidden = false;
  } finally { loading = false; }
}
$("refresh").addEventListener("click", refresh);
$("environment").addEventListener("change", renderRuns);
$("scenario").addEventListener("change", renderRuns);
$("comparison").addEventListener("change", () => { pickLatestComparison = false; renderRuns(); });
$("comparison-mode").addEventListener("change", () => {
  $("comparison").value = "individual";
  pickLatestComparison = true;
  renderRuns();
});
$("show-exploratory").addEventListener("click", () => {
  $("comparison-mode").value = "exploratory";
  pickLatestComparison = true;
  renderRuns();
});
refresh();
setInterval(refresh, 10000);
setInterval(refreshBenchmarkStatus, 2000);
