# -*- coding: utf-8 -*-
"""
service.py
A lightweight quantile service for VERL 0.2.0-dev (GRPO reward-time usage).

Updates in this version:
- Logging: add RotatingFileHandler . Env vars:
    QUANTILE_LOG_FILE (default ./quantile_service.log)
    QUANTILE_LOG_MAX_BYTES (default 50MB)
    QUANTILE_LOG_BACKUPS (default 5)
- Flush print: print TAIL 20 (newest->oldest) instead of HEAD 20.
- Observation: auto-dump MAIN snapshot every 20 flushes to JSON.
- New op: 'dump_main' to return all rows in MAIN queue.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional, Any, Dict

import torch
from multiprocessing.connection import Listener, Client
from logging.handlers import RotatingFileHandler

# -----------------------------
# Constants & Config
# -----------------------------
N: int = 3                             #测评技术指标的维度
MAX_LENGTH: int = 2048                 #队列长度
INIT: int = 128                        #初始放入长队长度
MAX_DIGITS: int = 1                    #保留小数点后位数


DEFAULT_QUERY_TIMEOUT_S: float = 2.0   # client-side per request timeout
DEFAULT_IO_TIMEOUT_S: float = 3.0      # server-side connection poll timeout

# -----------------------------
# Logging (控制台 + 滚动文件)
# -----------------------------
logger = logging.getLogger("quantile_service")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")

# 确保控制台 handler 存在
_has_console = any(
    isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    for h in logger.handlers
)
if not _has_console:
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)

# 确保文件 handler（旋转日志）存在
_has_file = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
if not _has_file:
    log_path = os.getenv("QUANTILE_LOG_FILE", "./quantile_service.log")
    max_bytes = int(os.getenv("QUANTILE_LOG_MAX_BYTES", str(50 * 1024 * 1024)))  # 50 MB
    backups = int(os.getenv("QUANTILE_LOG_BACKUPS", "5"))
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        _fh = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backups)
        _fh.setLevel(logging.INFO)
        _fh.setFormatter(_fmt)
        logger.addHandler(_fh)
        logger.info(f"[Log] File logging to {log_path} (maxBytes={max_bytes}, backups={backups})")
    except Exception as e:
        logger.warning(f"[Log] Failed to init file logger: {e}")

# Optional: restrict torch threads to reduce contention
try:
    torch.set_num_threads(1)
except Exception:
    pass

# -----------------------------
# Core Service (data structure)
# -----------------------------
class QuantileCore:
    """
    Core data manager with:
    - main_queue: FIFO torch.FloatTensor [L, 3]
    - buffer: list of 3-float tuples
    - sorted_snapshot: immutable per-dim sorted tensors for ECDF queries
    """

    def __init__(self):
        init_tensor = torch.zeros((INIT, N), dtype=torch.float32)
        self._main: torch.Tensor = init_tensor.clone().contiguous()
        self._buffer: List[Tuple[float, float, float]] = []

        self._snapshot_per_dim: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._snapshot_lock = threading.Lock()
        self._buffer_lock = threading.Lock()

        self._rebuild_snapshot(self._main)

    # ---------- internal helpers ----------
    def _rebuild_snapshot(self, main: torch.Tensor) -> None:
        s0 = torch.sort(main[:, 0]).values.contiguous()
        s1 = torch.sort(main[:, 1]).values.contiguous()
        s2 = torch.sort(main[:, 2]).values.contiguous()
        with self._snapshot_lock:
            self._snapshot_per_dim = (s0, s1, s2)

    # ---------- functional APIs ----------
    def query_quantile(self, x: Sequence[float]) -> Tuple[float, float, float]:
        try:
            if self._snapshot_per_dim is None:
                return (0.0, 0.0, 0.0)
            s0, s1, s2 = self._snapshot_per_dim  # lock-free read
            if s0.numel() == 0:
                return (0.0, 0.0, 0.0)

            x0, x1, x2 = float(x[0]), float(x[1]), float(x[2])

            c0 = int(torch.searchsorted(s0, torch.tensor(x0, dtype=s0.dtype), right=True))
            c1 = int(torch.searchsorted(s1, torch.tensor(x1, dtype=s1.dtype), right=True))
            c2 = int(torch.searchsorted(s2, torch.tensor(x2, dtype=s2.dtype), right=True))
            n = float(s0.numel())

            r0 = round(min(max(c0 / n, 0.0), 1.0), MAX_DIGITS)
            r1 = round(min(max(c1 / n, 0.0), 1.0), MAX_DIGITS)
            r2 = round(min(max(c2 / n, 0.0), 1.0), MAX_DIGITS)
            return (r0, r1, r2)
        except Exception as e:
            logger.exception(f"[QuantileCore] query_quantile error: {e}")
            return (0.0, 0.0, 0.0)

    def enqueue_batch(self, points: Iterable[Sequence[float]]) -> int:
        try:
            cnt = 0
            with self._buffer_lock:
                for p in points:
                    if p is None or len(p) != N:
                        continue
                    self._buffer.append((float(p[0]), float(p[1]), float(p[2])))
                    cnt += 1
            return cnt
        except Exception as e:
            logger.exception(f"[QuantileCore] enqueue_batch error: {e}")
            return 0

    def flush(self) -> Dict[str, int]:
        try:
            with self._buffer_lock:
                if len(self._buffer) == 0:
                    return {"added": 0, "dropped": 0, "new_len": int(self._main.shape[0])}
                buf_tensor = torch.tensor(self._buffer, dtype=torch.float32)
                self._buffer.clear()

            new_main = torch.cat([self._main, buf_tensor], dim=0)

            dropped = 0
            if new_main.shape[0] > MAX_LENGTH:
                dropped = new_main.shape[0] - MAX_LENGTH
                new_main = new_main[-MAX_LENGTH:, :]

            self._main = new_main.contiguous()
            self._rebuild_snapshot(self._main)

            return {"added": int(buf_tensor.shape[0]), "dropped": int(dropped), "new_len": int(self._main.shape[0])}
        except Exception as e:
            logger.exception(f"[QuantileCore] flush error: {e}")
            return {"added": 0, "dropped": 0, "new_len": int(self._main.shape[0])}

    # ---------- persistence ----------
    def save(self, save_dir: str) -> Dict[str, Any]:
        """
        Save to <save_dir>/state.pt with keys:
        - version: int
        - main: FloatTensor [L,3] (cpu)
        - buffer: FloatTensor [B,3] (cpu)
        - timestamp: float
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            with self._buffer_lock:
                main_cpu = self._main.detach().cpu().contiguous()
                if len(self._buffer) > 0:
                    buffer_cpu = torch.tensor(self._buffer, dtype=torch.float32)
                else:
                    buffer_cpu = torch.empty((0, N), dtype=torch.float32)

            payload = {
                "version": 1,
                "main": main_cpu,
                "buffer": buffer_cpu,
                "timestamp": time.time(),
            }
            path = os.path.join(save_dir, "state.pt")
            torch.save(payload, path)
            return {"ok": True, "path": path, "main_len": int(main_cpu.shape[0]), "buffer_len": int(buffer_cpu.shape[0])}
        except Exception as e:
            logger.exception(f"[QuantileCore] save error: {e}")
            return {"ok": False, "error": "save_failed"}

    def load(self, save_dir: str) -> Dict[str, Any]:
        """
        Load from <save_dir>/state.pt; replace main/buffer and rebuild snapshot.
        """
        try:
            path = os.path.join(save_dir, "state.pt")
            if not os.path.exists(path):
                return {"ok": False, "error": f"no_state:{path}"}
            payload = torch.load(path, map_location="cpu")
            main = payload.get("main", None)
            buffer = payload.get("buffer", None)
            if not isinstance(main, torch.Tensor) or main.ndim != 2 or main.shape[1] != N:
                return {"ok": False, "error": "bad_main"}
            if not isinstance(buffer, torch.Tensor) or buffer.ndim != 2 or buffer.shape[1] != N:
                return {"ok": False, "error": "bad_buffer"}

            main = main.to(dtype=torch.float32, copy=True).contiguous()
            buffer_list: List[Tuple[float, float, float]] = []
            if buffer.numel() > 0:
                buffer = buffer.to(dtype=torch.float32, copy=True).contiguous()
                for i in range(buffer.shape[0]):
                    buffer_list.append((float(buffer[i, 0]), float(buffer[i, 1]), float(buffer[i, 2])))

            with self._buffer_lock:
                self._main = main
                self._buffer = buffer_list
                self._rebuild_snapshot(self._main)

            return {"ok": True, "main_len": int(self._main.shape[0]), "buffer_len": int(len(self._buffer))}
        except Exception as e:
            logger.exception(f"[QuantileCore] load error: {e}")
            return {"ok": False, "error": "load_failed"}

    # ---------- stats / views ----------
    def main_mean_var(self) -> Dict[str, Any]:
        """Per-dim mean/var for MAIN queue only."""
        try:
            if self._main.numel() == 0:
                return {"ok": True, "mean": [0.0, 0.0, 0.0], "var": [0.0, 0.0, 0.0], "main_len": 0, "buffer_len": 0}
            m = self._main.to(dtype=torch.float64)
            mean = m.mean(dim=0)
            var = m.var(dim=0, unbiased=False)
            return {
                "ok": True,
                "mean": [float(mean[0]), float(mean[1]), float(mean[2])],
                "var": [float(var[0]), float(var[1]), float(var[2])],
                "main_len": int(self._main.shape[0]),
                "buffer_len": int(len(self._buffer)),
            }
        except Exception as e:
            logger.exception(f"[QuantileCore] main_mean_var error: {e}")
            return {"ok": True, "mean": [0.0, 0.0, 0.0], "var": [0.0, 0.0, 0.0], "main_len": 0, "buffer_len": 0}

    def main_head(self, k: int = 20) -> List[List[float]]:
        """Return first k rows (oldest->newest)."""
        try:
            if self._main.numel() == 0:
                return []
            k = max(0, min(int(k), int(self._main.shape[0])))
            return self._main[:k, :].detach().cpu().tolist()
        except Exception:
            return []

    def main_tail(self, k: int = 20) -> List[List[float]]:
        """Return last k rows (newest->oldest)."""
        try:
            if self._main.numel() == 0:
                return []
            k = max(0, min(int(k), int(self._main.shape[0])))
            tail = self._main[-k:, :].detach().cpu().tolist()
            tail.reverse()  # newest -> oldest
            return tail
        except Exception:
            return []

    def main_all(self) -> List[List[float]]:
        """Return all rows in MAIN (oldest->newest)."""
        try:
            if self._main.numel() == 0:
                return []
            return self._main.detach().cpu().tolist()
        except Exception:
            return []

# -----------------------------
# Server (multi-client)
# -----------------------------
@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 50070
    authkey: bytes = b"secret"
    io_timeout_s: float = DEFAULT_IO_TIMEOUT_S
    save_dir: str = "./quantile_state"
    resume: bool = False


class QuantileServiceServer:
    """
    TCP server using multiprocessing.connection.Listener.
    Ops:
      - ping
      - query: {'x': [x1,x2,x3]} -> [r1,r2,r3]
      - enqueue: {'points': [[...], ...]} -> {'added': int}
      - flush -> {'added','dropped','new_len','flush_count'}
      - save: {'path': optional} -> {'path','main_len','buffer_len'}
      - load: {'path': optional} -> {'main_len','buffer_len'}
      - stats -> {'mean':[3], 'var':[3], 'main_len', 'buffer_len', 'flush_count'}
      - dump_main -> {'main': [[...],[...], ...], 'main_len': int}
    """

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.core = QuantileCore()
        self.flush_count: int = 0  # always reset to 0 on init (even when resume)

        # optional resume
        if self.cfg.resume:
            res = self.core.load(self.cfg.save_dir)
            if not res.get("ok", False):
                logger.warning(f"[QuantileService] resume failed or not found: {res}")
            else:
                logger.info(f"[QuantileService] resumed: {res}")
        self._listener: Optional[Listener] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        address = (self.cfg.host, self.cfg.port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(address)
            except OSError as e:
                raise RuntimeError(f"Port {address} is already in use or not bindable: {e}")

        self._listener = Listener(address=address, authkey=self.cfg.authkey)
        logger.info(f"[QuantileService] Listening on {self.cfg.host}:{self.cfg.port} | save_dir={self.cfg.save_dir} | resume={self.cfg.resume}")

        self._accept_thread = threading.Thread(target=self._accept_loop, name="accept_loop", daemon=True)
        self._accept_thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            if self._listener is not None:
                self._listener.close()
        except Exception:
            pass
        logger.info("[QuantileService] Stopped.")

    def _accept_loop(self):
        assert self._listener is not None
        while not self._stop_event.is_set():
            try:
                conn = self._listener.accept()
                t = threading.Thread(target=self._client_handler, args=(conn,), daemon=True)
                t.start()
            except (OSError, EOFError):
                if not self._stop_event.is_set():
                    logger.warning("[QuantileService] Listener closed unexpectedly.")
                break
            except Exception as e:
                logger.exception(f"[QuantileService] accept error: {e}")
                time.sleep(0.1)

    def _client_handler(self, conn):
        try:
            while not self._stop_event.is_set():
                if not conn.poll(self.cfg.io_timeout_s):
                    continue
                req = conn.recv()
                resp = self._handle_req(req)
                try:
                    conn.send(resp)
                except Exception as e:
                    logger.exception(f"[QuantileService] send error: {e}")
                    break
        except (EOFError, OSError):
            pass
        except Exception as e:
            logger.exception(f"[QuantileService] client handler error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _handle_req(self, req: Dict[str, Any]) -> Dict[str, Any]:
        req_id = req.get("req_id")
        op = req.get("op", "")
        try:
            if op == "ping":
                return {"ok": True, "data": "pong", "req_id": req_id}

            if op == "query":
                x = req.get("x", None)
                if not isinstance(x, (list, tuple)) or len(x) != N:
                    return {"ok": True, "data": [0.0, 0.0, 0.0], "req_id": req_id}
                r = self.core.query_quantile(x)
                return {"ok": True, "data": [r[0], r[1], r[2]], "req_id": req_id}

            if op == "enqueue":
                points = req.get("points", None)
                if not isinstance(points, (list, tuple)) or len(points) == 0:
                    return {"ok": True, "data": {"added": 0}, "req_id": req_id}
                added = self.core.enqueue_batch(points)
                return {"ok": True, "data": {"added": int(added)}, "req_id": req_id}

            if op == "flush":
                stats = self.core.flush()
                # 先累加全局计数器
                self.flush_count += 1
                stats["flush_count"] = int(self.flush_count)

                if self.flush_count % 20 == 0:
                    self._dump_main_snapshot()

                # 统计 mean/var（MAIN 队列）
                s = self.core.main_mean_var()
                mean = s.get("mean", [0.0, 0.0, 0.0])
                var  = s.get("var",  [0.0, 0.0, 0.0])
                main_len = s.get("main_len", 0)

                # 取 MAIN 队列尾部 20 行（FIFO: 最新在前 -> 最旧）
                tail128 = self.core.main_tail(128)

                # 打印 flush 信息
                try:
                    logger.info(
                        "[FLUSH] #%d | added=%d dropped=%d new_len=%d | main_len=%d | "
                        "mean=(%.6f, %.6f, %.6f) var=(%.6f, %.6f, %.6f)",
                        self.flush_count,
                        int(stats.get("added", 0)),
                        int(stats.get("dropped", 0)),
                        int(stats.get("new_len", 0)),
                        int(main_len),
                        float(mean[0]), float(mean[1]), float(mean[2]),
                        float(var[0]),  float(var[1]),  float(var[2]),
                    )
                    if tail128:
                        logger.info("[FLUSH] main tail (up to 128, newest->oldest):")
                        # 计算真实下标（main_len-1, main_len-2, ...）
                        for i, row in enumerate(tail128):
                            idx = max(0, main_len - 1 - i)
                            x0 = float(row[0]) if len(row) > 0 else 0.0
                            x1 = float(row[1]) if len(row) > 1 else 0.0
                            x2 = float(row[2]) if len(row) > 2 else 0.0
                            logger.info("  [%04d] %.6f  %.6f  %.6f", idx, x0, x1, x2)
                    else:
                        logger.info("[FLUSH] main tail is empty.")
                except Exception:
                    pass

                return {"ok": True, "data": stats, "req_id": req_id}

            if op == "save":
                path = req.get("path", self.cfg.save_dir)
                res = self.core.save(path)
                if not res.get("ok", False):
                    return {"ok": False, "error": res.get("error", "save_failed"), "req_id": req_id}
                return {"ok": True, "data": res, "req_id": req_id}

            if op == "load":
                path = req.get("path", self.cfg.save_dir)
                res = self.core.load(path)
                if not res.get("ok", False):
                    return {"ok": False, "error": res.get("error", "load_failed"), "req_id": req_id}
                return {"ok": True, "data": res, "req_id": req_id}

            if op == "stats":
                s = self.core.main_mean_var()
                s["flush_count"] = int(self.flush_count)
                return {"ok": True, "data": s, "req_id": req_id}

            if op == "dump_main":
                arr = self.core.main_all()
                return {"ok": True, "data": {"main": arr, "main_len": len(arr)}, "req_id": req_id}

            return {"ok": False, "error": f"Unknown op: {op}", "req_id": req_id}
        except Exception as e:
            logger.exception(f"[QuantileService] handle_req error: {e}")
            if op == "query":
                return {"ok": True, "data": [0.0, 0.0, 0.0], "req_id": req_id}
            return {"ok": False, "error": "internal_error", "req_id": req_id}

    def _dump_main_snapshot(self) -> None:
        """Persist MAIN queue snapshot as JSON for inspection."""
        try:
            arr = self.core.main_all()
            dump_len = len(arr)
            dump_dir = os.path.join(self.cfg.save_dir, "observe_snapshots")
            os.makedirs(dump_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"observe_flush_{self.flush_count:06d}_{timestamp}.json"
            path = os.path.join(dump_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"flush_count": int(self.flush_count), "main": arr}, f)
            logger.info(
                f"[OBSERVE] Auto-dumped MAIN snapshot ({dump_len} rows) -> {path}"
            )
        except Exception as e:
            logger.warning(f"[OBSERVE] Failed to dump snapshot on flush #{self.flush_count}: {e}")

# -----------------------------
# Client
# -----------------------------
class QuantileServiceClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 50070, authkey: bytes = b"secret",
                 timeout_s: float = DEFAULT_QUERY_TIMEOUT_S):
        self.address = (host, port)
        self.authkey = authkey
        self.timeout_s = timeout_s
        self._conn: Optional[Client] = None
        self._req_counter = 0
        self._lock = threading.Lock()

    def connect(self):
        if self._conn is not None:
            return
        self._conn = Client(self.address, authkey=self.authkey)

    def close(self):
        try:
            if self._conn is not None:
                self._conn.close()
        finally:
            self._conn = None

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._conn is None:
            raise RuntimeError("Client not connected")
        try:
            with self._lock:
                self._req_counter += 1
                payload["req_id"] = f"req-{self._req_counter}"
                self._conn.send(payload)
                if not self._conn.poll(self.timeout_s):
                    return {"ok": False, "timeout": True}
                resp = self._conn.recv()
                return resp
        except Exception as e:
            logger.exception(f"[QuantileClient] request error: {e}")
            return {"ok": False, "error": "client_exception"}

    def ping(self) -> bool:
        resp = self._request({"op": "ping"})
        return bool(resp.get("ok") and resp.get("data") == "pong")

    def query(self, x: Sequence[float]) -> Tuple[float, float, float]:
        try:
            resp = self._request({"op": "query", "x": [float(x[0]), float(x[1]), float(x[2])]})
            if not resp.get("ok", False):
                return (0.0, 0.0, 0.0)
            data = resp.get("data", [0.0, 0.0, 0.0])
            if not (isinstance(data, (list, tuple)) and len(data) == 3):
                return (0.0, 0.0, 0.0)
            return (float(data[0]), float(data[1]), float(data[2]))
        except Exception:
            return (0.0, 0.0, 0.0)

    def enqueue_batch(self, points: Iterable[Sequence[float]]) -> int:
        pts = []
        for p in points:
            try:
                if p is None or len(p) != N:
                    continue
                pts.append([float(p[0]), float(p[1]), float(p[2])])
            except Exception:
                continue
        if not pts:
            return 0
        resp = self._request({"op": "enqueue", "points": pts})
        if not resp.get("ok", False):
            return 0
        data = resp.get("data", {})
        return int(data.get("added", 0))

    def flush(self) -> Dict[str, int]:
        resp = self._request({"op": "flush"})
        if not resp.get("ok", False):
            return {"added": 0, "dropped": 0, "new_len": 0, "flush_count": 0}
        data = resp.get("data", {})
        return {
            "added": int(data.get("added", 0)),
            "dropped": int(data.get("dropped", 0)),
            "new_len": int(data.get("new_len", 0)),
            "flush_count": int(data.get("flush_count", 0)),
        }

    # -------- convenience helpers --------
    def save(self, path: Optional[str] = None) -> Dict[str, Any]:
        payload = {"op": "save"}
        if path:
            payload["path"] = path
        resp = self._request(payload)
        return resp if resp.get("ok", False) else {"ok": False, "error": resp.get("error", "save_failed")}

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        payload = {"op": "load"}
        if path:
            payload["path"] = path
        resp = self._request(payload)
        return resp if resp.get("ok", False) else {"ok": False, "error": resp.get("error", "load_failed")}

    def stats(self) -> Dict[str, Any]:
        resp = self._request({"op": "stats"})
        if not resp.get("ok", False):
            return {"ok": False, "error": resp.get("error", "stats_failed")}
        return resp["data"]

    def dump_main(self) -> List[List[float]]:
        """Fetch all rows from MAIN (oldest->newest)."""
        resp = self._request({"op": "dump_main"})
        if not resp.get("ok", False):
            return []
        data = resp.get("data", {})
        return data.get("main", []) or []

# -----------------------------
# CLI Entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50070)
    parser.add_argument("--authkey", type=str, default="secret")
    parser.add_argument("--save_dir", type=str, default="./log")
    parser.add_argument("--resume", action="store_true", help="Load from save_dir on startup")
    args = parser.parse_args()

    cfg = ServerConfig(
        host=args.host,
        port=args.port,
        authkey=args.authkey.encode("utf-8"),
        save_dir=args.save_dir,
        resume=bool(args.resume),
    )
    server = QuantileServiceServer(cfg)
    server.start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
    finally:
        server.stop()

if __name__ == "__main__":
    main()
