"""Small HTTP utilities shared by the server, probes and benchmark."""

import http.client
import json
import os
import ssl
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

MAX_BODY = 2 * 1024 * 1024


def open_url(url, data=None, headers=None, timeout=5, context=None):
    # Cluster and loopback requests must not silently go through host HTTP proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context))
    request = urllib.request.Request(url, data=data, headers=headers or {})
    return opener.open(request, timeout=timeout)


def get_json(url, timeout=5, headers=None):
    with open_url(url, timeout=timeout, headers=headers) as response:
        return json.load(response)


def upstream_connection(origin, timeout):
    url = urlsplit(origin)
    connection = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    return connection(url.hostname, url.port, timeout=timeout)


def upstream_headers(config):
    headers = {"Content-Type": "application/json"}
    if config.api_key_env:
        key = os.environ.get(config.api_key_env)
        if not key:
            raise ValueError(f"upstream key environment variable is missing: {config.api_key_env}")
        headers["Authorization"] = f"Bearer {key}"
    return headers


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 connection closing also delimits SSE without chunk rewriting.
    protocol_version = "HTTP/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(65)

    def log_message(self, *_):
        pass

    def reply(self, status, body, content_type="application/json", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (headers or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(body)

    def error(self, status, message):
        self.reply(status, {"error": {"message": message, "type": "kubeai_error"}})

    def read_json(self):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("chunked request bodies are unsupported")
        size = int(self.headers.get("Content-Length", "0"))
        if size <= 0 or size > MAX_BODY:
            raise ValueError(f"request body must be 1..{MAX_BODY} bytes")
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise ValueError("incomplete request body")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value


def validate_chat(body):
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise ValueError("model must be a nonempty string")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str) or "content" not in message:
            raise ValueError("each message requires role and content")
    budget = body.get("max_completion_tokens", body.get("max_tokens", 64))
    if type(budget) is not int or not 1 <= budget <= 131072:
        raise ValueError("output token budget must be an integer from 1 to 131072")
    if "stream" in body and type(body["stream"]) is not bool:
        raise ValueError("stream must be a boolean")
    return budget
