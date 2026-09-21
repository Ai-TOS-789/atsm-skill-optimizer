#!/usr/bin/env python3
"""ATSM Federation Protocol: multi-instance collaboration over LAN.

Allows multiple ATSM instances to discover each other, sync skill data,
delegate tasks, and reach consensus on skill rankings via UDP multicast.

Usage:
    python3 atsm_federation.py start [--secret SECRET] [--group ADDR] [--port PORT]
    python3 atsm_federation.py peers
    python3 atsm_federation.py sync
    python3 atsm_federation.py delegate peer_id "task description"
    python3 atsm_federation.py stats
"""

import argparse
import base64
import hashlib
import json
import logging
import os
import pickle
import queue
import random
import signal
import socket
import struct
import sys
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MULTICAST_GROUP = "239.255.42.99"
DEFAULT_PORT = 4299
PEER_TIMEOUT = 30.0       # seconds before a peer is considered stale
PING_INTERVAL = 5.0       # seconds between PING broadcasts
GOSSIP_INTERVAL = 15.0    # seconds between sync gossip rounds
BUFFER_SIZE = 65535

DATA_DIR = Path(__file__).parent.parent / "data"
FEDERATION_DIR = DATA_DIR / "federation"
PEERS_FILE = FEDERATION_DIR / "peers.json"
SYNC_LOG = FEDERATION_DIR / "sync_log.jsonl"
PID_FILE = FEDERATION_DIR / "federation.pid"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("atsm-federation")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def derive_fernet_key(shared_secret: str) -> bytes:
    """Derive a URL-safe base64 32-byte key from an arbitrary shared secret."""
    digest = hashlib.sha256(shared_secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def get_instance_id() -> str:
    """Return a stable per-instance identifier."""
    id_file = FEDERATION_DIR / "instance.id"
    if id_file.exists():
        return id_file.read_text().strip()
    inst_id = uuid.uuid4().hex[:12]
    id_file.write_text(inst_id)
    return inst_id


def ensure_dirs():
    FEDERATION_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------
MSG_PING = "PING"
MSG_SYNC_REQUEST = "SYNC_REQUEST"
MSG_SYNC_RESPONSE = "SYNC_RESPONSE"
MSG_TASK_DELEGATE = "TASK_DELEGATE"
MSG_TASK_RESULT = "TASK_RESULT"
MSG_CONSENSUS_VOTE = "CONSENSUS_VOTE"

ALL_MSG_TYPES = {
    MSG_PING, MSG_SYNC_REQUEST, MSG_SYNC_RESPONSE,
    MSG_TASK_DELEGATE, MSG_TASK_RESULT, MSG_CONSENSUS_VOTE,
}


def make_message(msg_type: str, payload: dict, instance_id: str) -> dict:
    return {
        "type": msg_type,
        "from": instance_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Federation Node
# ---------------------------------------------------------------------------
class FederationNode:
    """UDP-multicast federation node for ATSM collaboration."""

    def __init__(self, secret: str, group: str = DEFAULT_MULTICAST_GROUP,
                 port: int = DEFAULT_PORT):
        ensure_dirs()
        self.instance_id = get_instance_id()
        self.key = derive_fernet_key(secret)
        self.fernet = Fernet(self.key)
        self.group = group
        self.port = port
        self.peers: dict[str, dict] = {}          # peer_id -> {addr, last_seen, ...}
        self.running = False
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self.task_results: dict[str, dict] = {}   # task_id -> result
        self.consensus_votes: dict[str, list] = defaultdict(list)  # topic -> [votes]

        # Load persisted peers
        self._load_peers()

    # ---- crypto ----
    def _encrypt(self, data: dict) -> bytes:
        raw = json.dumps(data).encode("utf-8")
        return self.fernet.encrypt(raw)

    def _decrypt(self, token: bytes) -> dict | None:
        try:
            raw = self.fernet.decrypt(token)
            return json.loads(raw)
        except Exception:
            return None

    # ---- socket setup ----
    def _create_socket(self) -> socket.socket:
        """Create a reusable UDP multicast socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        sock.bind(("", self.port))
        # Join multicast group
        mreq = struct.pack(
            "4sL",
            socket.inet_aton(self.group),
            socket.INADDR_ANY,
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(0.5)
        return sock

    # ---- send helpers ----
    def _send_mcast(self, msg: dict):
        """Encrypt and send to the multicast group."""
        token = self._encrypt(msg)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(token, (self.group, self.port))

    def _send_unicast(self, msg: dict, addr: tuple):
        """Encrypt and send to a specific address."""
        token = self._encrypt(msg)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(token, addr)

    # ---- peer management ----
    def _load_peers(self):
        if PEERS_FILE.exists():
            try:
                self.peers = json.loads(PEERS_FILE.read_text())
            except Exception:
                self.peers = {}

    def _save_peers(self):
        with self._lock:
            PEERS_FILE.write_text(json.dumps(self.peers, indent=2))

    def _purge_stale_peers(self):
        now = time.time()
        stale = [
            pid for pid, info in self.peers.items()
            if now - info.get("last_seen", 0) > PEER_TIMEOUT
        ]
        for pid in stale:
            del self.peers[pid]
        if stale:
            self._save_peers()
        return stale

    # ---- message handlers ----
    def _handle_ping(self, msg: dict, addr: tuple):
        peer_id = msg["from"]
        if peer_id == self.instance_id:
            return
        with self._lock:
            self.peers[peer_id] = {
                "addr": f"{addr[0]}:{addr[1]}",
                "last_seen": time.time(),
                "info": msg["payload"].get("info", {}),
            }
            self._save_peers()

    def _handle_sync_request(self, msg: dict, addr: tuple):
        """Respond with our local ATSM priors & outcome log."""
        peer_id = msg["from"]
        if peer_id == self.instance_id:
            return
        response_msg = make_message(MSG_SYNC_RESPONSE, self._gather_local_data(), self.instance_id)
        self._send_unicast(response_msg, addr)
        self._log_sync("received_request", peer_id)

    def _handle_sync_response(self, msg: dict, addr: tuple):
        peer_id = msg["from"]
        data = msg["payload"]
        self._merge_peer_data(peer_id, data)
        self._log_sync("received_data", peer_id)

    def _handle_task_delegate(self, msg: dict, addr: tuple):
        peer_id = msg["from"]
        if peer_id == self.instance_id:
            return
        task_desc = msg["payload"].get("task", "")
        task_id = msg["payload"].get("task_id", uuid.uuid4().hex[:8])
        # Simulate task execution (real integration would call ATSM rank/exec)
        result = self._execute_task(task_desc)
        result_msg = make_message(MSG_TASK_RESULT, {
            "task_id": task_id,
            "result": result,
            "status": "completed",
        }, self.instance_id)
        self._send_unicast(result_msg, addr)

    def _handle_task_result(self, msg: dict, addr: tuple):
        task_id = msg["payload"].get("task_id", "")
        self.task_results[task_id] = msg["payload"]

    def _handle_consensus_vote(self, msg: dict, addr: tuple):
        topic = msg["payload"].get("topic", "")
        vote = msg["payload"].get("vote")
        if vote is not None:
            self.consensus_votes[topic].append({
                "from": msg["from"],
                "vote": vote,
                "timestamp": msg["timestamp"],
            })

    # ---- data helpers ----
    def _gather_local_data(self) -> dict:
        """Gather ATSM priors, outcomes, and skills metadata."""
        priors = {}
        priors_file = DATA_DIR / "atsm_priors.json"
        if priors_file.exists():
            try:
                priors = json.loads(priors_file.read_text())
            except Exception:
                pass
        outcomes = []
        db_file = DATA_DIR / "atsm_db.jsonl"
        if db_file.exists():
            with db_file.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            outcomes.append(json.loads(line))
                        except Exception:
                            pass
        return {
            "instance_id": self.instance_id,
            "priors": priors,
            "outcomes": outcomes[-100:],  # last 100
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _merge_peer_data(self, peer_id: str, data: dict):
        """Merge incoming peer data into local store (simple append for now)."""
        merge_file = FEDERATION_DIR / "merged_data.jsonl"
        entry = {
            "peer_id": peer_id,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        with merge_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _execute_task(self, task_desc: str) -> str:
        """Simulated task execution. Replace with real ATSM integration."""
        return f"Simulated execution of: {task_desc}"

    def _log_sync(self, action: str, peer_id: str):
        with SYNC_LOG.open("a") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "peer": peer_id,
            }) + "\n")

    # ---- background threads ----
    def _ping_loop(self):
        """Periodically broadcast PING to discover peers."""
        info = {"version": "1.0", "atsm": True, "hostname": socket.gethostname()}
        msg = make_message(MSG_PING, {"info": info}, self.instance_id)
        while not self._stop_evt.is_set():
            try:
                self._send_mcast(msg)
            except Exception as e:
                log.debug("ping send failed: %s", e)
            self._stop_evt.wait(PING_INTERVAL)

    def _cleanup_loop(self):
        """Periodically remove stale peers."""
        while not self._stop_evt.is_set():
            self._purge_stale_peers()
            self._stop_evt.wait(PING_INTERVAL * 2)

    def _receive_loop(self):
        """Listen for incoming multicast messages."""
        sock = self._create_socket()
        log.info("Listening on %s:%d (instance %s)", self.group, self.port, self.instance_id)
        while not self._stop_evt.is_set():
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break

            msg = self._decrypt(data)
            if msg is None or msg.get("type") not in ALL_MSG_TYPES:
                continue

            mtype = msg["type"]
            if mtype == MSG_PING:
                self._handle_ping(msg, addr)
            elif mtype == MSG_SYNC_REQUEST:
                self._handle_sync_request(msg, addr)
            elif mtype == MSG_SYNC_RESPONSE:
                self._handle_sync_response(msg, addr)
            elif mtype == MSG_TASK_DELEGATE:
                self._handle_task_delegate(msg, addr)
            elif mtype == MSG_TASK_RESULT:
                self._handle_task_result(msg, addr)
            elif mtype == MSG_CONSENSUS_VOTE:
                self._handle_consensus_vote(msg, addr)
        sock.close()

    # ---- public API ----
    def start(self):
        """Start the federation node (blocking)."""
        self.running = True
        PID_FILE.write_text(str(os.getpid()))
        log.info("ATSM Federation node starting (id=%s)", self.instance_id)

        t_recv = threading.Thread(target=self._receive_loop, daemon=True)
        t_ping = threading.Thread(target=self._ping_loop, daemon=True)
        t_cleanup = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._threads = [t_recv, t_ping, t_cleanup]

        for t in self._threads:
            t.start()

        def _shutdown(sig, frame):
            log.info("Shutting down...")
            self.stop()
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        try:
            while self.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        self._stop_evt.set()
        for t in self._threads:
            t.join(timeout=3)
        if PID_FILE.exists():
            PID_FILE.unlink()
        log.info("Stopped.")

    def get_peers(self) -> dict:
        self._purge_stale_peers()
        with self._lock:
            return dict(self.peers)

    def request_sync(self):
        """Broadcast a SYNC_REQUEST to all peers."""
        msg = make_message(MSG_SYNC_REQUEST, {"request": True}, self.instance_id)
        self._send_mcast(msg)

    def delegate_task(self, peer_id: str, task: str) -> str:
        """Send TASK_DELEGATE to a specific peer. Returns task_id."""
        with self._lock:
            peer = self.peers.get(peer_id)
        if not peer:
            raise ValueError(f"Unknown peer: {peer_id}")
        addr_str = peer["addr"]
        host, port_str = addr_str.rsplit(":", 1)
        addr = (host, int(port_str))

        task_id = uuid.uuid4().hex[:8]
        msg = make_message(MSG_TASK_DELEGATE, {
            "task": task,
            "task_id": task_id,
        }, self.instance_id)
        self._send_unicast(msg, addr)
        return task_id

    def get_stats(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "peers_count": len(self.peers),
            "peers": list(self.peers.keys()),
            "task_results": len(self.task_results),
            "consensus_topics": {
                k: len(v) for k, v in self.consensus_votes.items()
            },
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_start(args):
    secret = args.secret or os.environ.get("ATSM_SECRET", "atsm-default-secret-change-me")
    node = FederationNode(secret=secret, group=args.group, port=args.port)
    node.start()


def cmd_peers(args):
    """Check for peers by briefly joining the multicast group and listening."""
    ensure_dirs()
    instance_id = get_instance_id()
    # Load persisted peers
    if PEERS_FILE.exists():
        peers = json.loads(PEERS_FILE.read_text())
    else:
        peers = {}

    # Ping for new peers
    secret = args.secret or os.environ.get("ATSM_SECRET", "atsm-default-secret-change-me")
    key = derive_fernet_key(secret)
    fernet = Fernet(key)

    info = {"version": "1.0", "atsm": True}
    msg = make_message(MSG_PING, {"info": info}, instance_id)
    token = fernet.encrypt(json.dumps(msg).encode("utf-8"))

    # Send ping
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(token, (args.group, args.port))

    # Listen briefly
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        pass
    sock.bind(("", args.port))
    mreq = struct.pack("4sL", socket.inet_aton(args.group), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(3.0)

    deadline = time.time() + 3.0
    while time.time() < deadline:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
        except socket.timeout:
            continue
        try:
            raw = fernet.decrypt(data)
            peer_msg = json.loads(raw)
        except Exception:
            continue
        if peer_msg.get("type") == MSG_PING and peer_msg.get("from") != instance_id:
            pid = peer_msg["from"]
            peers[pid] = {
                "addr": f"{addr[0]}:{addr[1]}",
                "last_seen": time.time(),
                "info": peer_msg["payload"].get("info", {}),
            }
    sock.close()

    # Persist
    PEERS_FILE.write_text(json.dumps(peers, indent=2))

    print(json.dumps({
        "instance_id": instance_id,
        "peers_count": len(peers),
        "peers": peers,
    }, indent=2))


def cmd_sync(args):
    """Initiate a sync round (broadcast SYNC_REQUEST, listen for responses)."""
    ensure_dirs()
    instance_id = get_instance_id()
    secret = args.secret or os.environ.get("ATSM_SECRET", "atsm-default-secret-change-me")
    key = derive_fernet_key(secret)
    fernet = Fernet(key)

    msg = make_message(MSG_SYNC_REQUEST, {"request": True}, instance_id)
    token = fernet.encrypt(json.dumps(msg).encode("utf-8"))

    # Send request
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(token, (args.group, args.port))

    # Listen for SYNC_RESPONSE
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        pass
    sock.bind(("", args.port))
    mreq = struct.pack("4sL", socket.inet_aton(args.group), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(3.0)

    responses = []
    deadline = time.time() + 3.0
    while time.time() < deadline:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
        except socket.timeout:
            continue
        try:
            raw = fernet.decrypt(data)
            peer_msg = json.loads(raw)
        except Exception:
            continue
        if peer_msg.get("type") == MSG_SYNC_RESPONSE and peer_msg.get("from") != instance_id:
            responses.append(peer_msg["payload"])
            # Merge
            merge_file = FEDERATION_DIR / "merged_data.jsonl"
            with merge_file.open("a") as f:
                f.write(json.dumps({
                    "peer_id": peer_msg["from"],
                    "received_at": datetime.now(timezone.utc).isoformat(),
                    "data": peer_msg["payload"],
                }) + "\n")
    sock.close()

    print(json.dumps({
        "instance_id": instance_id,
        "responses_received": len(responses),
        "responses": [{"instance_id": r.get("instance_id")} for r in responses],
    }, indent=2))


def cmd_delegate(args):
    """Delegate a task to a specific peer."""
    ensure_dirs()
    peer_id = args.peer_id
    task = args.task

    # Find peer address
    if PEERS_FILE.exists():
        peers = json.loads(PEERS_FILE.read_text())
    else:
        peers = {}
    if peer_id not in peers:
        print(json.dumps({"error": f"Unknown peer: {peer_id}"}))
        sys.exit(1)

    secret = args.secret or os.environ.get("ATSM_SECRET", "atsm-default-secret-change-me")
    key = derive_fernet_key(secret)
    fernet = Fernet(key)
    instance_id = get_instance_id()

    addr_str = peers[peer_id]["addr"]
    host, port_str = addr_str.rsplit(":", 1)
    addr = (host, int(port_str))

    task_id = uuid.uuid4().hex[:8]
    msg = make_message(MSG_TASK_DELEGATE, {"task": task, "task_id": task_id}, instance_id)
    token = fernet.encrypt(json.dumps(msg).encode("utf-8"))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(token, addr)

    print(json.dumps({
        "status": "delegated",
        "task_id": task_id,
        "to": peer_id,
        "task": task,
    }, indent=2))


def cmd_stats(args):
    """Show federation stats."""
    ensure_dirs()
    instance_id = get_instance_id()

    peers = {}
    if PEERS_FILE.exists():
        try:
            peers = json.loads(PEERS_FILE.read_text())
        except Exception:
            pass

    sync_count = 0
    if SYNC_LOG.exists():
        with SYNC_LOG.open() as f:
            for line in f:
                if line.strip():
                    sync_count += 1

    merged = 0
    merge_file = FEDERATION_DIR / "merged_data.jsonl"
    if merge_file.exists():
        with merge_file.open() as f:
            for line in f:
                if line.strip():
                    merged += 1

    print(json.dumps({
        "instance_id": instance_id,
        "peers_count": len(peers),
        "peers": {k: v.get("addr") for k, v in peers.items()},
        "sync_events": sync_count,
        "merged_entries": merged,
    }, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ATSM Federation Protocol")
    sub = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = sub.add_parser("start", help="Run the federation node")
    p_start.add_argument("--secret", default=os.environ.get("ATSM_SECRET", ""),
                         help="Shared secret (or set ATSM_SECRET env var)")
    p_start.add_argument("--group", default=DEFAULT_MULTICAST_GROUP)
    p_start.add_argument("--port", type=int, default=DEFAULT_PORT)

    # peers
    p_peers = sub.add_parser("peers", help="Discover and list peers")
    p_peers.add_argument("--secret", default=os.environ.get("ATSM_SECRET", ""))
    p_peers.add_argument("--group", default=DEFAULT_MULTICAST_GROUP)
    p_peers.add_argument("--port", type=int, default=DEFAULT_PORT)

    # sync
    p_sync = sub.add_parser("sync", help="Request data sync from peers")
    p_sync.add_argument("--secret", default=os.environ.get("ATSM_SECRET", ""))
    p_sync.add_argument("--group", default=DEFAULT_MULTICAST_GROUP)
    p_sync.add_argument("--port", type=int, default=DEFAULT_PORT)

    # delegate
    p_del = sub.add_parser("delegate", help="Delegate a task to a peer")
    p_del.add_argument("peer_id", help="Target peer ID")
    p_del.add_argument("task", help="Task description")
    p_del.add_argument("--secret", default=os.environ.get("ATSM_SECRET", ""))
    p_del.add_argument("--group", default=DEFAULT_MULTICAST_GROUP)
    p_del.add_argument("--port", type=int, default=DEFAULT_PORT)

    # stats
    p_stats = sub.add_parser("stats", help="Show federation statistics")
    p_stats.add_argument("--secret", default=os.environ.get("ATSM_SECRET", ""))

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "peers":
        cmd_peers(args)
    elif args.command == "sync":
        cmd_sync(args)
    elif args.command == "delegate":
        cmd_delegate(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
