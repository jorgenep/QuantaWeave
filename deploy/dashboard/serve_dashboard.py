"""Dependency-free live viewer for QuantaWeave training runs, plus a system-telemetry tab.

Serves the static dashboard (index.html, deploy/dashboard/), whichever run's metrics.jsonl
under --runs-dir is requested (or, with none specified, whichever was written to most recently --
the active run, by definition), and a live CPU/RAM/GPU telemetry feed. Everything is read-only:
this never touches a training process, its files, or the GPU beyond querying free memory.

  python3 deploy/dashboard/serve_dashboard.py --runs-dir artifacts/outputs --port 8090
"""

import argparse
import glob
import http.server
import json
import re
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path


def resolve_interface_ip(interface: str) -> str:
    """The current IPv4 address of a named network interface (e.g. tailscale0, wg0, eth0).

    Binding to this instead of 0.0.0.0 means the dashboard is only reachable over that interface --
    e.g. only over a Tailscale or other VPN tunnel, never a bare LAN IP or public one. Resolved at
    startup, not hardcoded, so it keeps working if a VPN reassigns its address later; if the
    interface renames or the address changes while running, restart the service to pick it up.
    """
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", interface],
                              capture_output=True, text=True, timeout=5, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise SystemExit(f"--bind-interface {interface}: could not query it ({e}). "
                          f"Is the interface name right? Check with `ip -4 addr show`.")
    match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", out.stdout)
    if not match:
        raise SystemExit(f"--bind-interface {interface}: no IPv4 address found on that interface "
                          f"(is it up? check with `ip -4 addr show dev {interface}`).")
    return match.group(1)

DASHBOARD_DIR = Path(__file__).resolve().parent
HWMON_GLOB = "/sys/class/drm/card0/device/hwmon/hwmon*"
VRAM_REFRESH_EVERY = 30  # seconds; the torch query below takes ~2-3s, so it runs far less often than the 1s loop


def discover_runs(runs_dir: Path) -> list[dict]:
    runs = []
    if not runs_dir.is_dir():
        return runs
    for metrics_path in runs_dir.glob("*/metrics.jsonl"):
        try:
            stat = metrics_path.stat()
        except OSError:
            continue
        runs.append({"name": metrics_path.parent.name, "mtime": stat.st_mtime, "size": stat.st_size})
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def read_proc_stat() -> dict[str, int]:
    with open("/proc/stat") as f:
        first = f.readline().split()
    fields = [int(x) for x in first[1:]]
    labels = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
    return dict(zip(labels, fields))


def read_meminfo() -> dict[str, int]:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            value = rest.strip().split()[0]
            out[key] = int(value) * 1024  # kB -> bytes
    return out


def read_hwmon() -> dict:
    temps, fan, energy = {}, None, {}
    for hwmon in glob.glob(HWMON_GLOB):
        try:
            if (Path(hwmon) / "name").read_text().strip() != "xe":
                continue
        except OSError:
            continue
        for temp_input in glob.glob(f"{hwmon}/temp*_input"):
            label_path = temp_input.replace("_input", "_label")
            try:
                label = Path(label_path).read_text().strip()
                temps[label] = int(Path(temp_input).read_text().strip()) / 1000.0
            except OSError:
                pass
        fan_path = Path(hwmon) / "fan1_input"
        if fan_path.exists():
            try:
                fan = int(fan_path.read_text().strip())
            except OSError:
                pass
        for energy_input in glob.glob(f"{hwmon}/energy*_input"):
            label_path = energy_input.replace("_input", "_label")
            try:
                label = Path(label_path).read_text().strip()
                energy[label] = int(Path(energy_input).read_text().strip())
            except OSError:
                pass
    return {"temps": temps, "fan_rpm": fan, "energy_uj": energy}


def read_xe_engine_cycles() -> tuple[dict[str, int], dict[str, int]]:
    """Per-engine cumulative GPU cycles, summed across every client of the xe driver."""
    used: dict[str, int] = {}
    total: dict[str, int] = {}
    for fdinfo in glob.glob("/proc/*/fdinfo/*"):
        try:
            with open(fdinfo) as f:
                text = f.read()
        except OSError:
            continue
        if "drm-driver:\txe" not in text:
            continue
        for m in re.finditer(r"drm-cycles-(\w+):\s*(\d+)", text):
            used[m.group(1)] = used.get(m.group(1), 0) + int(m.group(2))
        for m in re.finditer(r"drm-total-cycles-(\w+):\s*(\d+)", text):
            total[m.group(1)] = max(total.get(m.group(1), 0), int(m.group(2)))
    return used, total


def read_nvtop() -> dict:
    try:
        out = subprocess.run(["nvtop", "-s"], capture_output=True, timeout=2, text=True)
        rows = json.loads(out.stdout)
        return rows[0] if rows else {}
    except Exception:
        return {}


class SystemMonitor:
    """Background sampler: cheap sysfs/procfs reads every second, the one expensive VRAM
    query far less often. HTTP handlers just read `self.snapshot` -- never block on I/O."""

    def __init__(self, python_bin: Path):
        self.python_bin = python_bin
        self.snapshot: dict = {}
        self.lock = threading.Lock()
        self._prev_stat = read_proc_stat()
        self._prev_cycles, self._prev_cycles_total = read_xe_engine_cycles()
        self._prev_energy = read_hwmon()["energy_uj"]
        self._prev_time = time.time()
        self._vram = {"free_bytes": None, "total_bytes": None, "checked_at": None}
        self._tick_count = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _cpu_percent(self) -> float:
        cur = read_proc_stat()
        deltas = {k: cur[k] - self._prev_stat[k] for k in cur}
        self._prev_stat = cur
        total = sum(deltas.values())
        idle = deltas["idle"] + deltas["iowait"]
        return 0.0 if total <= 0 else max(0.0, min(100.0, 100.0 * (total - idle) / total))

    def _gpu_engines(self, dt: float) -> dict[str, float]:
        cur, cur_total = read_xe_engine_cycles()
        out = {}
        for k, tot in cur_total.items():
            dtot = tot - self._prev_cycles_total.get(k, tot)
            dused = cur.get(k, 0) - self._prev_cycles.get(k, 0)
            out[k] = 0.0 if dtot <= 0 else max(0.0, min(100.0, 100.0 * dused / dtot))
        self._prev_cycles, self._prev_cycles_total = cur, cur_total
        return out

    def _power_watts(self, dt: float) -> dict[str, float]:
        cur = read_hwmon()["energy_uj"]
        out = {}
        for k, v in cur.items():
            prev = self._prev_energy.get(k)
            if prev is not None and dt > 0:
                out[k] = (v - prev) / dt / 1e6
        self._prev_energy = cur
        return out

    def _refresh_vram(self) -> None:
        try:
            proc = subprocess.run(
                [str(self.python_bin), "-c", "import torch;f,t=torch.xpu.mem_get_info();print(f,t)"],
                capture_output=True, timeout=15, text=True,
            )
            free_s, total_s = proc.stdout.strip().split()
            self._vram = {"free_bytes": int(free_s), "total_bytes": int(total_s), "checked_at": time.time()}
        except Exception:
            pass  # keep the last good reading; XPU may be transiently busy

    def _loop(self) -> None:
        while True:
            now = time.time()
            dt = now - self._prev_time
            self._prev_time = now

            cpu = self._cpu_percent()
            mem = read_meminfo()
            hwmon = read_hwmon()
            engines = self._gpu_engines(dt)
            power = self._power_watts(dt)
            nvtop = read_nvtop()

            self._tick_count += 1
            if self._tick_count % VRAM_REFRESH_EVERY == 1:
                threading.Thread(target=self._refresh_vram, daemon=True).start()

            with self.lock:
                self.snapshot = {
                    "timestamp": now,
                    "cpu_percent": cpu,
                    "load_avg": list(__import__("os").getloadavg()),
                    "mem": {
                        "total_bytes": mem.get("MemTotal"),
                        "available_bytes": mem.get("MemAvailable"),
                        "used_bytes": mem.get("MemTotal", 0) - mem.get("MemAvailable", 0),
                        "swap_total_bytes": mem.get("SwapTotal"),
                        "swap_used_bytes": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0),
                    },
                    "gpu": {
                        "name": nvtop.get("device_name"),
                        "clock": nvtop.get("gpu_clock"),
                        "fan_rpm": hwmon["fan_rpm"],
                        "temps_c": hwmon["temps"],
                        "engine_util_percent": engines,
                        "power_watts": power,
                        "vram": self._vram,
                    },
                }
            time.sleep(1.0)


def make_handler(runs_dir: Path, monitor: SystemMonitor):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            pass  # keep stdout quiet; this runs unattended for days

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib method name
            from urllib.parse import urlsplit, parse_qs

            parsed = urlsplit(self.path)
            path, query = parsed.path, parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                body = (DASHBOARD_DIR / "index.html").read_bytes()
                self._send(200, body, "text/html; charset=utf-8")
                return

            if path == "/api/runs":
                body = json.dumps(discover_runs(runs_dir)).encode()
                self._send(200, body, "application/json")
                return

            if path == "/api/system":
                with monitor.lock:
                    body = json.dumps(monitor.snapshot).encode()
                self._send(200, body, "application/json")
                return

            if path == "/metrics.jsonl":
                runs = discover_runs(runs_dir)
                requested = query.get("run", [None])[0]
                target = requested or (runs[0]["name"] if runs else None)
                if target is None:
                    self._send(200, b"", "text/plain; charset=utf-8")
                    return
                metrics_path = runs_dir / target / "metrics.jsonl"
                body = metrics_path.read_bytes() if metrics_path.exists() else b""
                self._send(200, body, "text/plain; charset=utf-8")
                return

            self._send(404, b"not found", "text/plain; charset=utf-8")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True, help="directory of run subfolders, each with a metrics.jsonl")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable),
                         help="python with torch+xpu installed, used only for the periodic VRAM query")
    parser.add_argument("--host", default="0.0.0.0",
                         help="address to bind; ignored if --bind-interface is given. Default 0.0.0.0 is "
                              "every interface, including any public one -- fine on a trusted LAN, but "
                              "consider --bind-interface for anything internet-facing.")
    parser.add_argument("--bind-interface",
                         help="bind only to this interface's current address instead of --host, e.g. "
                              "'tailscale0' (Tailscale), 'wg0' (WireGuard), or your LAN NIC's name -- "
                              "resolved at startup so it tracks address changes across restarts. "
                              "Use `ip -4 addr show` to list interface names on this host.")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    host = resolve_interface_ip(args.bind_interface) if args.bind_interface else args.host

    monitor = SystemMonitor(args.python_bin)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((host, args.port), make_handler(args.runs_dir, monitor)) as httpd:
        via = f"{args.bind_interface} at " if args.bind_interface else ""
        print(f"dashboard serving on http://{host}:{args.port} ({via}runs: {args.runs_dir})")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
