#!/usr/bin/env python3
"""E5 host: drive the CAN bench, run the IDS + TreeSHAP online, and label ground truth.

The PC is the stand-in for the vehicle's central computer. It
  - replays a ROAD ambient capture onto the bus through the CANable (background traffic of
    the vehicle the model was trained on; the paper shows the model does not transfer to
    another vehicle's traffic, so the bench must reuse ROAD's own);
  - commands the MCU nodes over serial (firmware in firmware/) to fabricate, masquerade or
    fuzz;
  - classifies every frame with the paper's XGBoost IDS and explains it with TreeSHAP,
    timing each verdict against the 10 ms per-ID deadline, optionally while the GPU runs the
    E1 perception loop (`--gpu-load`).

Subcommands
  selftest  online feature extractor == can_features.extract (must be exact), plus
            per-frame predict+TreeSHAP latency on a ROAD log. Needs no hardware.
  train     train the whole-bus IDS on ROAD captures and save it.
  run       execute a scenario on the bench and write capture, events, verdicts, summary.

Ground truth: own frames come back as python-can echoes (is_rx=False), so every frame the PC
replayed is background; frames received from others come from the MCU nodes, and are attack
frames when their ID is in the scenario's attack set and an attack interval is open. This
gives per-frame labels, not window labels: fabrication windows contain benign frames of the
same ID, and labelling them as attacks inflates the detection rate (the paper's per-frame
point).

Nothing here invents a number: every summary value is computed from the logged frames.
"""
import argparse, json, sys, time, threading, queue, statistics as st
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[2]  # repo root holds can_features.py, road_common.py
sys.path.insert(0, str(BENCH))
import can_features as cf  # noqa: E402

DEADLINE_MS = 10.0


class OnlineFeatures:
    """Incremental twin of can_features.extract: same state, one row per frame.

    `selftest` proves equality on a real capture; if extract changes, this must change too.
    """

    def __init__(self, window=0.5):
        self.window = window
        self.last_ts, self.last_pl = {}, {}
        self.deltas = defaultdict(lambda: deque(maxlen=cf.PERIOD_HISTORY))
        self.win = deque()
        self.id_win = defaultdict(deque)

    def row(self, ts, cid, pl):
        d = ts - self.last_ts[cid] if cid in self.last_ts else 0.0
        hist = self.deltas[cid]
        nominal = float(np.median(hist)) if len(hist) >= 5 else 0.0
        dev = (d - nominal) / nominal if nominal > 1e-9 else 0.0
        win = self.win
        while win and ts - win[0] > self.window:
            win.popleft()
        win.append(ts)
        iw = self.id_win[cid]
        while iw and ts - iw[0] > self.window:
            iw.popleft()
        iw.append(ts)
        prev = self.last_pl.get(cid)
        if prev is not None and len(prev) == len(pl):
            xor = [a ^ b for a, b in zip(prev, pl)]
            ham = sum(bin(v).count('1') for v in xor)
            chg = sum(1 for v in xor if v)
        else:
            ham = chg = 0
        r = np.array([d, dev, len(iw), len(win) / self.window, cf.entropy(pl), ham, chg, len(pl)]
                     + [pl[k] if k < len(pl) else 0 for k in range(8)], dtype=np.float32)
        self.last_ts[cid] = ts
        self.last_pl[cid] = pl
        hist.append(d)
        return r


def load_model(path):
    import xgboost as xgb
    m = xgb.XGBClassifier()
    m.load_model(str(path))
    m.get_booster().set_param({"nthread": 1})  # one core per verdict, as in the paper's cost runs
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    if meta.get("feats") != list(cf.FEATS):
        raise SystemExit("model was trained on a different feature list; retrain")
    return m, meta


def verdict(booster, row):
    """Score + TreeSHAP for one frame; returns (score, shap_vector, latency_ms)."""
    import xgboost as xgb
    t0 = time.perf_counter()
    dm = xgb.DMatrix(row.reshape(1, -1), feature_names=list(cf.FEATS))
    score = float(booster.predict(dm)[0])
    contrib = booster.predict(dm, pred_contribs=True)[0]
    return score, contrib, (time.perf_counter() - t0) * 1e3


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


# --------------------------------------------------------------------------- selftest
def cmd_selftest(a):
    msgs = cf.parse_log(a.log, limit=a.limit)
    if not msgs:
        raise SystemExit(f"no frames parsed from {a.log}")
    Xb, _ = cf.extract(msgs)
    on = OnlineFeatures()
    Xo = np.vstack([on.row(ts, cid, pl) for ts, cid, pl in msgs])
    diff = float(np.max(np.abs(Xb - Xo)))
    out = {"log": str(a.log), "frames": len(msgs), "max_abs_diff_online_vs_batch": diff,
           "equal": diff == 0.0}
    if a.model:
        model, _meta = load_model(a.model)
        b = model.get_booster()
        lat = [verdict(b, Xo[i])[2] for i in range(min(len(Xo), a.latency_n))]
        out["latency_ms"] = {"n": len(lat), "median": st.median(lat), "p99": q(lat, 0.99),
                             "deadline_ms": DEADLINE_MS,
                             "miss_frac": sum(x > DEADLINE_MS for x in lat) / len(lat)}
    print(json.dumps(out, indent=2))
    if not out["equal"]:
        raise SystemExit("online features differ from batch extract: fix before any run")


# --------------------------------------------------------------------------- train
def cmd_train(a):
    import road_common as rc
    X, y = rc.load_many(a.captures, only_target_id=False, verbose=True, precise=True,
                        strict=True)
    m = rc.make_model()
    m.fit(X, y)
    m.save_model(a.out)
    Path(a.out + ".meta.json").write_text(json.dumps(
        {"captures": a.captures, "rows": int(len(y)), "attack_frac": float(y.mean()),
         "feats": list(cf.FEATS), "labels": "per-frame (precise)", "scope": "whole bus"},
        indent=2))
    print(f"saved {a.out} ({len(y)} rows, {y.mean() * 100:.2f}% attack)")


# --------------------------------------------------------------------------- run
class Node:
    """One MCU on a serial port; every line is logged with the host clock."""

    def __init__(self, name, port, log):
        import serial
        self.name, self.log = name, log
        self.s = serial.Serial(port, 115200, timeout=0.1)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.alive = True
        self.t.start()

    def _read(self):
        while self.alive:
            raw = self.s.readline()
            if raw:
                self.log.put({"t": time.perf_counter(), "src": self.name,
                              "line": raw.decode(errors="replace").strip()})

    def send(self, cmd):
        self.log.put({"t": time.perf_counter(), "src": "host->" + self.name, "line": cmd})
        self.s.write((cmd + "\n").encode())

    def close(self):
        self.alive = False
        self.s.close()


def cmd_run(a):
    import can
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scen = json.loads(Path(a.scenario).read_text())
    any_id = "ANY" in scen["attack_ids"]  # fuzzing: every frame a node sends is an attack
    attack_ids = {int(x, 16) for x in scen["attack_ids"] if x != "ANY"}
    events = queue.Queue()
    nodes = {}
    for spec in (a.nodes or []):
        name, port = spec.split("=", 1)
        nodes[name] = Node(name, port, events)
    time.sleep(2.5)  # boards print READY after reset
    for n in nodes.values():
        n.send(f"BITRATE {a.bitrate}")

    bus = can.Bus(interface=a.interface, channel=a.channel, bitrate=a.bitrate * 1000,
                  receive_own_messages=True)
    bg = cf.parse_log(a.background, limit=a.bg_limit)
    if not bg:
        raise SystemExit("no background frames")
    suspended = {}  # id -> host time until which the replay omits it (masquerade victim)
    open_attacks = []  # (start, end, kind)
    stop = threading.Event()       # stops replay and reception
    stop_ids = threading.Event()   # stops the IDS worker, after reception has stopped
    t_start = time.perf_counter()

    def replay():
        t0 = bg[0][0]
        while not stop.is_set():
            base = time.perf_counter()
            for ts, cid, pl in bg:
                if stop.is_set():
                    return
                due = base + (ts - t0)
                # sleep only, never spin: a busy-wait holds the GIL and starves the IDS
                # thread, which would make the IDS look slower than it is. The replay
                # jitter this costs is measured from capture.log, not assumed.
                rem = due - time.perf_counter()
                if rem > 0:
                    time.sleep(rem)
                if suspended.get(cid, 0) > time.perf_counter():
                    continue
                try:
                    bus.send(can.Message(arbitration_id=cid, data=pl, is_extended_id=False))
                except can.CanError:
                    events.put({"t": time.perf_counter(), "src": "host", "line": "TX_ERROR"})

    rx = queue.Queue()
    received = [0]

    def receive():
        while not stop.is_set():
            m = bus.recv(timeout=0.1)
            if m is not None and not m.is_error_frame:
                received[0] += 1
                rx.put((time.perf_counter(), m))

    model, meta = load_model(a.model)
    booster = model.get_booster()
    feats = OnlineFeatures()
    rows = []      # one row per frame that got a verdict
    cap = []       # every frame seen, for capture.log
    watch = {int(x, 16) for x in a.watch_ids} if a.watch_ids else None

    def ids_worker():
        while not stop_ids.is_set():
            try:
                t_arr, m = rx.get(timeout=0.1)
            except queue.Empty:
                continue
            pl = bytes(m.data)
            r = feats.row(t_arr - t_start, m.arbitration_id, pl)  # state updates for all ids
            cap.append((t_arr - t_start, m.arbitration_id, pl.hex()))
            if watch is not None and m.arbitration_id not in watch:
                continue  # per-ID deadline view: verdicts only for the watched ids
            score, contrib, lat = verdict(booster, r)
            done = (time.perf_counter() - t_arr) * 1e3  # queueing + compute
            in_att = any(s <= t_arr <= e for s, e, _k in open_attacks)
            hit = any_id or m.arbitration_id in attack_ids
            label = int(m.is_rx and hit and in_att) if a.label_mode == "frame" \
                else int(in_att and hit)
            top = int(np.argmax(np.abs(contrib[:-1])))
            rows.append((t_arr - t_start, m.arbitration_id, int(m.is_rx), pl.hex(), score,
                         lat, done, label, cf.FEATS[top]))

    threads = [threading.Thread(target=f, daemon=True) for f in (replay, receive, ids_worker)]
    gpu = None
    if a.gpu_load:
        import subprocess, shlex
        gpu = subprocess.Popen(shlex.split(a.gpu_load))
    for t in threads:
        t.start()

    # Scenario: [{"at": s, "node": name|"host", "cmd": "...", "attack": "fab|masq|fuzz"?,
    #             "dur_ms": n?}]
    for step in sorted(scen["steps"], key=lambda s: s["at"]):
        while time.perf_counter() - t_start < step["at"]:
            time.sleep(0.005)
        now = time.perf_counter()
        if step["node"] == "host":
            verb, *args = step["cmd"].split()
            if verb == "SUSPEND":  # stop replaying an id (masquerade victim)
                suspended[int(args[0], 16)] = now + int(args[1]) / 1000
            events.put({"t": now, "src": "host", "line": step["cmd"]})
        else:
            nodes[step["node"]].send(step["cmd"])
        if step.get("attack"):
            open_attacks.append((now, now + step["dur_ms"] / 1000, step["attack"]))
    time.sleep(scen.get("tail_s", 5))
    stop.set()
    for t in threads[:2]:          # replay, receive
        t.join(timeout=5)
    stop_ids.set()
    threads[2].join(timeout=30)    # the IDS finishes the frame it is on, then stops
    backlog = rx.qsize()  # frames that arrived but never got a verdict
    for n in nodes.values():
        n.send("STAT")
    time.sleep(0.5)
    if gpu:
        gpu.terminate()
    bus.shutdown()
    for n in nodes.values():
        n.close()

    # ---- write everything, then summarise from what was written
    with open(out / "capture.log", "w") as f:
        for t, cid, hx in cap:
            f.write(f"({t:.6f}) can0 {cid:03X}#{hx.upper()}\n")
    with open(out / "verdicts.csv", "w") as f:
        f.write("t,id,is_rx,payload,score,compute_ms,arrival_to_verdict_ms,label,top_feature\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    ev = []
    while not events.empty():
        e = events.get()
        e["t"] -= t_start
        ev.append(e)
    (out / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev))
    comp = [r[5] for r in rows]
    tot = [r[6] for r in rows]
    y = np.array([r[7] for r in rows])
    s = np.array([r[4] for r in rows])
    pred = s >= a.threshold
    summary = {
        "frames_received": received[0], "frames_with_verdict": len(rows),
        "backlog_without_verdict": backlog, "frames_seen_by_ids": len(cap),
        "watch_ids": a.watch_ids,
        "frames": len(rows), "own_echo_frames": int(sum(1 for r in rows if not r[2])),
        "attack_frames": int(y.sum()), "label_mode": a.label_mode,
        "threshold": a.threshold, "model_meta": meta, "scenario": a.scenario,
        "interface": a.interface, "bitrate_kbps": a.bitrate, "gpu_load": a.gpu_load,
        "compute_ms": {"median": st.median(comp), "p99": q(comp, 0.99)} if comp else None,
        "arrival_to_verdict_ms": {"median": st.median(tot), "p99": q(tot, 0.99),
                                  "deadline_ms": DEADLINE_MS,
                                  "miss_frac": float(np.mean(np.array(tot) > DEADLINE_MS))}
        if tot else None,
        "recall": float(pred[y == 1].mean()) if y.sum() else None,
        "fpr": float(pred[y == 0].mean()) if (y == 0).any() else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if backlog:
        print(f"NOTE: {backlog} frames never got a verdict: the IDS did not keep up with the "
              "bus. Report this; it is a result, not a bug to hide.")
    if not summary["own_echo_frames"]:
        print("WARNING: no own echoes received; frame-level labels are not trustworthy "
              "with this interface (check receive_own_messages support).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("selftest")
    p.add_argument("--log", required=True)
    p.add_argument("--limit", type=int, default=50000)
    p.add_argument("--model")
    p.add_argument("--latency-n", type=int, default=2000)
    p.set_defaults(fn=cmd_selftest)
    p = sp.add_parser("train")
    p.add_argument("--captures", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_train)
    p = sp.add_parser("run")
    p.add_argument("--scenario", required=True)
    p.add_argument("--background", required=True, help="ROAD ambient .log to replay")
    p.add_argument("--bg-limit", type=int, default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--interface", default="slcan", help="slcan | gs_usb | virtual | socketcan")
    p.add_argument("--channel", required=True, help="e.g. COM5, can0, or a virtual name")
    p.add_argument("--bitrate", type=int, default=500, help="kbit/s")
    p.add_argument("--nodes", nargs="*", help="name=PORT, e.g. attacker=COM7 ecu=COM8")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--watch-ids", nargs="*", help="hex ids that get a verdict (default: all)")
    p.add_argument("--label-mode", choices=["frame", "window"], default="frame")
    p.add_argument("--gpu-load", help="command that keeps the GPU busy, e.g. the E1 inference loop")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
