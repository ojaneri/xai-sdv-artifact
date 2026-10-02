# E5 — Integrated bench: CAN IDS + TreeSHAP on the "central computer" while perception runs

The desktop PC stands in for the SDV central computer. It replays ROAD ambient traffic onto
a real CAN bus, MCU nodes inject attacks, and the paper's IDS classifies and explains frames
online against the 10 ms per-ID deadline, optionally while the GPU runs the E1 perception
loop. The question is the editor's: does the framework work **as one system**, with both
domains competing for the same computer?

## Hardware (all in the inventory; CANable arrives by 17 Oct)

```
PC (RTX 3060) ──USB── CANable 2.0 ──┐
                                    │ CAN_H / CAN_L twisted pair, 500 kbit/s
Nucleo-L476RG ─SPI─ MCP2515+TJA1050 ┤   120 Ω at EACH end of the bus (from the resistor kit;
ESP32-S3      ─SPI─ MCP2515+TJA1050 ┘   skip one if a module/CANable already terminates)
```

- **MCP2515 crystal:** read the can on each module (8.000 or 16.000 MHz) and set `MCP_OSC`
  in `firmware/platformio.ini` per board. A wrong value = wrong bitrate = error frames.
- **Power:** the TJA1050 needs 5 V. Its RX output into a 3.3 V MCU needs a divider or a
  3.3 V-tolerant pin. Nucleo D12 (MISO) is 5 V-tolerant; on the ESP32-S3 it is not, so add a
  divider on MISO or use a 3.3 V transceiver module. Check before powering.
- **Wiring:** SCK, MISO, MOSI and CS (pins in `platformio.ini`), with common GND across all
  boards and the CANable.
- **CANable firmware:** ships with slcan (`--interface slcan --channel COM?`). The
  candleLight firmware (`--interface gs_usb`) gives hardware timestamps and is preferred if
  you are willing to reflash it.

## Software

```
cd exp/E5/firmware
pio run -e nucleo_l476rg -t upload      # attacker by default in the scenarios
pio run -e esp32s3 -t upload            # optional second node
pio device monitor                      # expect: READY board=... bus=up
```

Then type `STAT`: `tec` and `rec` must stay at 0 on an idle, terminated bus.

Host environment: the repo's `requirements.txt` plus `python-can pyserial`, in a venv, with
the ROAD dataset in `data/road/` (`python download_road.py`).

## Procedure (each run writes to `exp/E5/results/<name>/`)

0. **Self-test (no hardware).** Proves the online features equal the paper's batch
   features, and gives per-frame IDS+TreeSHAP latency on this PC:
   `python exp/E5/bench_host.py selftest --log data/road/ambient/ambient_dyno_drive_basic_short.log --model exp/E5/work/ids.json`
   It must print `"equal": true`.
1. **Train the IDS** on the ROAD 0x0D0 captures (fabrication + masquerade, whole bus,
   per-frame labels):
   `python exp/E5/bench_host.py train --captures max_speedometer_attack_1 max_speedometer_attack_1_masquerade --out exp/E5/work/ids.json`
   Keep `max_speedometer_attack_2*` out of training: they are the offline held-out check.
2. **Clean baseline**, without and then with GPU load:
   `python exp/E5/bench_host.py run --scenario exp/E5/scenarios/clean_baseline.json --background data/road/ambient/ambient_dyno_drive_basic_long.log --model exp/E5/work/ids.json --interface slcan --channel COM5 --nodes attacker=COM7 --watch-ids 0D0 --out exp/E5/results/clean`
   Then add `--gpu-load "python exp/E1/<inference-loop>.py"`, with the E1 detector running in
   a loop, and use `--out exp/E5/results/clean_gpu`.
3. **Attacks**, each without and with `--gpu-load`: `fab_0d0.json`, `masq_0d0.json`,
   `fuzz.json`. For fuzzing, drop `--watch-ids` (any id can be the attack).
4. **Replay fidelity:** compare inter-arrival times of 0x0D0 in `capture.log` against the
   ROAD source (Windows timer jitter is real). Report it next to the results; if the jitter
   drives false positives on the clean baseline, that is a finding about PC-based replay,
   and the fix is to move the background replay onto a microcontroller.

## What each run reports (`summary.json`)

- frames received, frames with a verdict, and the **backlog without a verdict** (the IDS
  not keeping up is a result, not a bug);
- compute time and arrival-to-verdict time (median, p99) and the fraction missing the
  10 ms deadline;
- recall and FPR against **per-frame** ground truth: the PC's own echoes are background,
  frames from the MCU nodes on attack ids inside an attack interval are attacks.
  `--label-mode window` reproduces the window labels most CAN IDS papers use, for
  comparison only.

`events.jsonl` keeps every serial line and host action with host timestamps; `verdicts.csv`
keeps one row per verdict with its top TreeSHAP feature.

## Rules

Same as E1: no invented numbers, failures go into `NOTES.md`, small files only in git.
Commit `exp/E5/results/` and push, or zip it.
