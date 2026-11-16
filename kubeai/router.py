"""OpenAI-compatible chat proxy with streaming, probes, telemetry and dashboard."""

import hmac
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import __version__, discovery, endpoints, experiments, results
from .config import Config
from .metrics import Counters, label, signals
from .policy import Overloaded, Policy, Unavailable
from .transport import Handler, Server, get_json, open_url, upstream_connection, upstream_headers, validate_chat

LOG = logging.getLogger("kubeai")
STATIC = Path(__file__).with_name("static")


class Router:
    def __init__(self, config, config_path=None, results_dir="results"):
        self.policy = Policy(config)
        self.config_path = Path(config_path) if config_path else None
        self.config_mtime = self.config_path.stat().st_mtime_ns if self.config_path else None
        self.results_dir = Path(results_dir)
        self.counters = Counters()
        self.stop_event = threading.Event()
        self.thread = None
        self.discovery_at = 0.0
        self.discovered = []
        self.control_error = ""
        self.control_lock = threading.RLock()
        self.run_catalog = results.RunCatalog(self.results_dir)
        self.benchmarks = experiments.Benchmarks(self)

    def probe(self, replica):
        try:
            headers = upstream_headers(replica.config)
            models = get_json(replica.config.url.rstrip("/") + "/v1/models", timeout=self.policy.config.probe_timeout, headers=headers)
            available = {item["id"] for item in models["data"]}
            with self.policy.lock:
                replica.models = available
                replica.healthy_at = time.monotonic()
                replica.last_error = ""
        except Exception as error:
            with self.policy.lock:
                replica.healthy_at = 0
                replica.last_error = type(error).__name__
            return
        try:
            with open_url(replica.config.url.rstrip("/") + "/metrics", timeout=self.policy.config.probe_timeout, headers=headers) as response:
                data = signals(response.read(2 * 1024 * 1024).decode())
            with self.policy.lock:
                replica.metrics = data
                replica.metrics_at = time.monotonic() if data else 0.0
        except Exception:
            # Missing telemetry doesn't erase a still-fresh sample or model health.
            pass
        if replica.config.gpu_metrics_url:
            try:
                with open_url(replica.config.gpu_metrics_url, timeout=self.policy.config.probe_timeout) as response:
                    data = signals(response.read(1024 * 1024).decode())
                with self.policy.lock:
                    replica.gpu_metrics = {k: v for k, v in data.items() if k in ("memory_free", "memory_total")}
                    replica.gpu_metrics_at = time.monotonic() if replica.gpu_metrics else 0.0
            except Exception:
                pass

    def poll(self):
        with self.control_lock:
            replicas = self.reconcile()
        with ThreadPoolExecutor(max_workers=min(16, max(1, len(replicas)))) as pool:
            list(pool.map(self.probe, replicas))

    def reconcile(self):
        """Reload and reconcile under the control lock; probes run outside it."""
        if self.benchmarks.active:
            try:
                if self.config_path and self.config_path.stat().st_mtime_ns != self.config_mtime:
                    self.control_error = "Configuration file changed during the benchmark"
                    self.benchmarks.interrupt(self.control_error)
            except OSError:
                self.control_error = "Configuration file became unavailable during the benchmark"
                self.benchmarks.interrupt(self.control_error)
            with self.policy.lock:
                return list(self.policy.replicas.values())
        try:
            if self.config_path and self.config_path.stat().st_mtime_ns != self.config_mtime:
                config = Config.load(self.config_path)
                with self.policy.lock:
                    self.policy.config = config
                self.config_mtime = self.config_path.stat().st_mtime_ns
            config = self.policy.config
            if config.discovery:
                try:
                    self.discovered = discovery.endpoints(config.discovery)
                    self.discovery_at = time.monotonic()
                except Exception:
                    if time.monotonic() - self.discovery_at > config.stale_after:
                        self.discovered = []
                    raise
            else:
                self.discovered = []
            self.control_error = ""
        except Exception as error:
            self.control_error = type(error).__name__ + ": " + str(error)
            LOG.warning("configuration/discovery: %s", self.control_error)
        config = self.policy.config
        self.policy.reconcile([*config.replicas, *self.discovered])
        with self.policy.lock:
            return list(self.policy.replicas.values())

    def endpoint_document(self):
        with self.control_lock:
            return {**endpoints.read(self.config_path, self.policy.config), "locked": self.benchmarks.active}

    def apply_endpoints(self, values, revision, port):
        with self.control_lock:
            if self.benchmarks.active:
                raise endpoints.Conflict("A benchmark is running. Apply endpoints after it finishes.")
            config, mtime, view = endpoints.save(self.config_path, self.policy.config, values, revision, port)
            with self.policy.lock:
                self.policy.config = config
                self.policy.reconcile(config.replicas)
            self.config_mtime = mtime
            self.control_error = ""
        return view

    def start(self):
        self.poll()

        def loop():
            while not self.stop_event.wait(self.policy.config.poll_interval):
                self.poll()

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.benchmarks.stop()
        if self.thread:
            self.thread.join(timeout=5)

    def state(self):
        with self.control_lock:
            return {"version": __version__, "policy": self.policy.config.policy, "replicas": self.policy.snapshot(),
                    "control_error": self.control_error, "score_weights": self.policy.config.weights,
                    "configuration": asdict(self.policy.config),
                    "benchmark": {"active": self.benchmarks.active,
                                  "job_id": self.benchmarks.job["job_id"] if self.benchmarks.job else None}}

    def runs(self):
        return self.run_catalog.list()

    def metrics(self):
        lines = [self.counters.render()]
        for replica in self.policy.snapshot():
            tags = f'replica="{label(replica["name"])}"'
            for metric, value in (("inflight", replica["inflight"]), ("healthy", int(replica["healthy"])),
                                  ("metrics_fresh", int(replica["metrics_fresh"]))):
                lines.append(f"kubeai_replica_{metric}{{{tags}}} {value}\n")
        return "".join(lines)


def make_server(router, host="127.0.0.1", port=8080):
    class RouterHandler(Handler):
        def authorized(self):
            key = os.environ.get("KUBEAI_API_KEY", "")
            if key and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key):
                self.error(401, "invalid bearer token")
                return False
            return True

        def admin_authorized(self):
            if not self.authorized():
                return False
            try:
                host = self.headers.get("Host", "")
                hostname = urlsplit("http://" + host).hostname
                if not os.environ.get("KUBEAI_API_KEY") and not (
                    endpoints.loopback(self.client_address[0]) and endpoints.loopback(hostname)
                ):
                    self.error(403, "Operator actions without an API key are limited to loopback hosts")
                    return False
                origin = self.headers.get("Origin")
                if origin:
                    parsed = urlsplit(origin)
                    if parsed.scheme not in ("http", "https") or parsed.netloc.lower() != host.lower() or parsed.path not in ("", "/"):
                        self.error(403, "Operator actions require a same-origin request")
                        return False
                if self.headers.get("Sec-Fetch-Site") == "cross-site":
                    self.error(403, "Cross-site operator actions are not allowed")
                    return False
            except ValueError:
                self.error(403, "Invalid request origin")
                return False
            if self.headers.get_content_type() != "application/json":
                self.error(415, "Operator actions require application/json")
                return False
            return True

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/healthz":
                return self.reply(200, {"status": "ok"})
            if path == "/readyz":
                ready = any(r["healthy"] and r["models"] for r in router.policy.snapshot())
                return self.reply(200 if ready else 503, {"ready": ready})
            if path in ("/", "/app.js", "/style.css"):
                filename, content_type = {"/": ("index.html", "text/html; charset=utf-8"),
                                          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                                          "/style.css": ("style.css", "text/css; charset=utf-8")}[path]
                return self.reply(200, (STATIC / filename).read_bytes(), content_type,
                                  {"Content-Security-Policy": "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"})
            if not self.authorized():
                return
            if path == "/metrics":
                return self.reply(200, router.metrics(), "text/plain; version=0.0.4; charset=utf-8")
            if path == "/v1/models":
                models = sorted({m for r in router.policy.snapshot() if r["healthy"] for m in r["models"]})
                return self.reply(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "kubeai", "created": 0} for m in models]})
            if path == "/api/state":
                return self.reply(200, router.state())
            if path == "/api/benchmark":
                return self.reply(200, router.benchmarks.status())
            if path == "/api/benchmark/options":
                try:
                    return self.reply(200, router.benchmarks.options())
                except (OSError, ValueError, TypeError):
                    return self.error(500, "Could not load benchmark inputs")
            if path == "/api/endpoints":
                try:
                    return self.reply(200, router.endpoint_document())
                except (OSError, ValueError, TypeError) as error:
                    return self.error(409, "Cannot load saved endpoint configuration: " + type(error).__name__)
            if path == "/api/runs":
                return self.reply(200, {"runs": router.runs()})
            if path.startswith("/api/runs/"):
                detail = router.run_catalog.details(path[len("/api/runs/"):])
                if detail is None:
                    return self.error(404, "Run not found or its identifier is ambiguous")
                return self.reply(200, detail)
            if path == "/api/comparisons":
                modes = parse_qs(urlsplit(self.path).query, keep_blank_values=True).get("mode", ["recorded"])
                if len(modes) != 1 or modes[0] not in ("recorded", "exploratory"):
                    return self.error(400, "Comparison mode must be recorded or exploratory")
                return self.reply(200, {"mode": modes[0], "groups": results.comparisons(router.runs(), mode=modes[0])})
            self.error(404, "unknown endpoint")

        def do_POST(self):
            if urlsplit(self.path).path == "/api/benchmark":
                if not self.admin_authorized():
                    return
                try:
                    body = self.read_json()
                    host, port = self.server.server_address[:2]
                    if host == "0.0.0.0":
                        host = "127.0.0.1"
                    result = router.benchmarks.start(body, f"http://{host}:{port}")
                    return self.reply(202, result)
                except endpoints.Conflict as error:
                    return self.error(409, str(error))
                except (ValueError, TypeError, KeyError, UnicodeError) as error:
                    return self.error(400, str(error))
                except OSError:
                    return self.error(500, "Could not create benchmark files; check results directory permissions")
            if urlsplit(self.path).path == "/api/endpoints/test":
                if not self.admin_authorized():
                    return
                try:
                    body = self.read_json()
                    replica = endpoints.parse_replica(body.get("replica"), self.server.server_port)
                    return self.reply(200, endpoints.test_connection(replica))
                except (ValueError, TypeError, UnicodeError) as error:
                    return self.error(400, str(error))
            if urlsplit(self.path).path != "/v1/chat/completions":
                return self.error(404, "unknown endpoint")
            if not self.authorized():
                return
            try:
                body = self.read_json()
                tokens = validate_chat(body)
            except (ValueError, UnicodeError) as error:
                return self.error(400, str(error))
            request_id = uuid.uuid4().hex
            started = time.monotonic()
            policy_name = router.policy.config.policy
            try:
                replica, decision = router.policy.reserve(body["model"], tokens)
                policy_name = decision["policy"]
            except (Overloaded, Unavailable) as error:
                status = 429 if isinstance(error, Overloaded) else 503
                router.counters.record(policy_name, "none", status, time.monotonic() - started)
                return self.error(status, str(error))
            connection = None
            status, failed, sent = 502, False, False
            try:
                connection = upstream_connection(replica.config.url, router.policy.config.upstream_timeout)
                headers = upstream_headers(replica.config)
                headers["X-Request-ID"] = request_id
                connection.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers=headers)
                response = connection.getresponse()
                status = response.status
                failed = status >= 500 or status in (404, 429)
                self.send_response(status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("X-KubeAI-Replica", replica.config.name)
                self.send_header("X-KubeAI-Policy", policy_name)
                self.send_header("X-Request-ID", request_id)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if response.getheader("Retry-After"):
                    self.send_header("Retry-After", response.getheader("Retry-After"))
                self.end_headers()
                sent = True
                # read1 forwards available bytes immediately, preserving client TTFT.
                while chunk := response.read1(16384):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                status = 499
            except Exception as error:
                failed, status = True, 502
                if not sent:
                    self.error(502, "upstream connection failed: " + type(error).__name__)
            finally:
                if connection:
                    connection.close()
                duration = time.monotonic() - started
                router.policy.release(replica, failed)
                router.counters.record(policy_name, replica.config.name, status, duration)
                # No prompts, bearer tokens, or model output in the decision log.
                LOG.info(json.dumps({"request_id": request_id, "status": status, "duration_seconds": duration, **decision}))

        def do_PUT(self):
            if urlsplit(self.path).path != "/api/endpoints":
                return self.error(404, "unknown endpoint")
            if not self.admin_authorized():
                return
            try:
                body = self.read_json()
                result = router.apply_endpoints(body.get("replicas"), body.get("revision"), self.server.server_port)
                return self.reply(200, result)
            except (endpoints.Conflict, endpoints.ReadOnly) as error:
                return self.error(409, str(error))
            except (ValueError, TypeError, UnicodeError) as error:
                return self.error(400, str(error))
            except OSError:
                return self.error(500, "Could not save endpoint configuration; check file permissions")

    return Server((host, port), RouterHandler)
