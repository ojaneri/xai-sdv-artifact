#!/usr/bin/env python3
"""Fetch the ROAD CAN intrusion dataset from Zenodo.

The dataset is NOT redistributed with this code. It is 557 MB compressed
(3.0 GB extracted), published by Oak Ridge National Laboratory under CC BY 4.0:

    Verma, Iannacone, Bridges, Hollifield, Kay, Combs.
    "ROAD: The Real ORNL Automotive Dynamometer Controller Area Network
    Intrusion Detection Dataset." arXiv:2012.14600, 2020.
    DOI 10.13139/ORNLNCCS/1728694 -- https://zenodo.org/records/10462796

Cite it if you use it.

Usage:
    python download_road.py            # download, verify, extract
    python download_road.py --check    # verify an existing copy only

Verification is by byte count against the Zenodo record: a truncated download
that still extracts would silently produce a shorter capture and a wrong result.
"""
import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD = '10462796'
API = f'https://zenodo.org/api/records/{RECORD}'
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
TARGET = DATA / 'road'
EXPECTED_ATTACKS = 34
EXPECTED_AMBIENT = 13


def record_metadata():
    """Fetch the Zenodo record metadata as a dict."""
    with urllib.request.urlopen(API, timeout=60) as r:
        return json.load(r)


def check(verbose=True):
    """Is an extracted, complete copy already present?"""
    atk, amb = TARGET / 'attacks', TARGET / 'ambient'
    meta = atk / 'capture_metadata.json'
    if not (atk.is_dir() and amb.is_dir() and meta.is_file()):
        if verbose:
            print(f"  not present under {TARGET.relative_to(ROOT)}")
        return False
    n_atk = len(list(atk.glob('*.log')))
    n_amb = len(list(amb.glob('*.log')))
    ok = n_atk >= EXPECTED_ATTACKS - 1 and n_amb >= EXPECTED_AMBIENT - 1
    if verbose:
        print(f"  attacks: {n_atk} logs (expected ~{EXPECTED_ATTACKS})")
        print(f"  ambient: {n_amb} logs (expected ~{EXPECTED_AMBIENT})")
        print(f"  metadata: {meta.relative_to(ROOT)}")
        print("  OK" if ok else "  INCOMPLETE -- re-run without --check")
    return ok


def progress(done, total):
    """Render a download progress bar on one line."""
    if not total:
        return
    pct = done / total * 100
    bar = '#' * int(pct / 2.5)
    sys.stdout.write(f"\r  [{bar:<40}] {pct:5.1f}%  {done/1e6:.0f}/{total/1e6:.0f} MB")
    sys.stdout.flush()


def download():
    """Download, verify by byte count, and extract the dataset."""
    meta = record_metadata()
    files = meta.get('files', [])
    if len(files) != 1:
        raise SystemExit(f"expected 1 file in the Zenodo record, found {len(files)}")
    entry = files[0]
    url = entry['links']['self']
    size = entry['size']
    DATA.mkdir(parents=True, exist_ok=True)
    zpath = DATA / entry['key']

    lic = (meta.get('metadata', {}).get('license') or {}).get('id', '?')
    print(f"ROAD dataset -- Zenodo record {RECORD}, licence {lic}")
    print(f"  {entry['key']}: {size/1e6:.1f} MB")

    if zpath.is_file() and zpath.stat().st_size == size:
        print("  archive already present and complete")
    else:
        done = 0
        with urllib.request.urlopen(url, timeout=120) as r, open(zpath, 'wb') as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                progress(done, size)
        print()
        got = zpath.stat().st_size
        if got != size:
            zpath.unlink()
            raise SystemExit(
                f"size mismatch: got {got} bytes, record says {size}. "
                "A truncated archive would extract into short captures and a "
                "wrong result, so the partial file was removed.")
        print(f"  verified: {got} bytes matches the record")

    print("  extracting (macOS resource forks skipped)...")
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if m.startswith('__MACOSX/') or Path(m).name.startswith('._'):
                continue
            z.extract(m, DATA)
    zpath.unlink()
    print(f"  extracted to {TARGET.relative_to(ROOT)} "
          f"({shutil.disk_usage(DATA).used and ''}"
          f"{sum(f.stat().st_size for f in TARGET.rglob('*') if f.is_file())/1e9:.1f} GB)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true',
                    help='verify an existing copy without downloading')
    args = ap.parse_args()
    print("Checking for an existing copy:")
    present = check()
    if args.check:
        sys.exit(0 if present else 1)
    if present:
        print("Nothing to do. Use --check to re-verify.")
        sys.exit(0)
    download()
    print("\nVerifying:")
    sys.exit(0 if check() else 1)
