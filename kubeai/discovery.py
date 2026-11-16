"""In-cluster EndpointSlice discovery using short-lived service-account tokens."""

import ipaddress
import json
import os
import ssl
from pathlib import Path
from urllib.parse import quote, urlencode

from .config import ReplicaConfig
from .transport import open_url


def endpoints(settings):
    root = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    if ":" in host:
        host = f"[{host}]"
    url = f"https://{host}:{port}/apis/discovery.k8s.io/v1/namespaces/{quote(settings['namespace'], safe='')}/endpointslices?"
    url += urlencode({"labelSelector": f"kubernetes.io/service-name={settings['service']}"})
    context = ssl.create_default_context(cafile=str(root / "ca.crt"))
    headers = {"Authorization": "Bearer " + (root / "token").read_text().strip()}
    with open_url(url, timeout=2, headers=headers, context=context) as response:
        return parse_slices(json.load(response), settings)


def parse_slices(data, settings):
    found = {}
    for item in data.get("items", []):
        ports = [p["port"] for p in item.get("ports", []) if p.get("port") and p.get("name") == settings.get("port_name", "http")]
        if not ports:
            continue
        for endpoint in item.get("endpoints", []):
            conditions = endpoint.get("conditions", {})
            if conditions.get("ready") is not True or conditions.get("terminating"):
                continue
            for address in endpoint.get("addresses", []):
                ip = ipaddress.ip_address(address)
                authority = f"[{ip}]" if ip.version == 6 else str(ip)
                name = f"{settings['service']}-{ip}-{ports[0]}"
                found[name] = ReplicaConfig.parse({**settings.get("replica", {}), "name": name, "url": f"http://{authority}:{ports[0]}"})
    return list(found.values())
