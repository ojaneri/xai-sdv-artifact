#!/usr/bin/env python3
"""Fetch the HCRL Car-Hacking dataset.

Not redistributed here. Published by the Hacking and Countermeasure Research Lab
(Korea University) at

    https://ocslab.hksecurity.net/Datasets/car-hacking-dataset

The page states: "For academic purposes, we are happy to release our datasets.
If you want to use our dataset for your experiment, please cite our paper."
So CITE THEM:

    H. M. Song, J. Woo, and H. K. Kim, "In-vehicle network intrusion detection
    using deep convolutional neural network," Vehicular Communications, vol. 21,
    100198, 2020.

    E. Seo, H. M. Song, and H. K. Kim, "GIDS: GAN based Intrusion Detection
    System for In-Vehicle Network," 16th Annual Conf. on Privacy, Security and
    Trust (PST), IEEE, 2018.

NOTE: the authors state that Table 2 of the PST paper contains errors, fixed in
the arXiv version. Use the arXiv numbers when comparing against them.

Traffic captured over OBD-II from a real Hyundai Sonata: DoS, fuzzy, drive-gear
spoofing and RPM spoofing, plus an attack-free capture. ~953 MB compressed.

Unlike ROAD, every frame carries its own label (R = normal, T = injected), so no
label derivation is needed.

Usage:
    python download_hcrl.py            # download, verify, extract
    python download_hcrl.py --check    # verify an existing copy only
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://www.dropbox.com/scl/fo/9rwsf9pclhvv9xxloojom/"
       "AF7JeRW893grZkigkulkAHk?rlkey=3h6zamu3kc262lrnipu5qden8&dl=1")
ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'data' / 'hcrl'
EXPECTED = ['DoS_dataset.csv', 'Fuzzy_dataset.csv',
            'gear_dataset.csv', 'RPM_dataset.csv']
MIN_BYTES = 150_000_000       # each attack CSV is 180-230 MB


def check(verbose=True):
    """Is a complete extracted copy already present?"""
    missing, short = [], []
    for name in EXPECTED:
        p = TARGET / name
        if not p.is_file():
            missing.append(name)
        elif p.stat().st_size < MIN_BYTES:
            short.append(f"{name} ({p.stat().st_size/1e6:.0f} MB)")
    ok = not missing and not short
    if verbose:
        if missing:
            print(f"  missing: {', '.join(missing)}")
        if short:
            print(f"  truncated: {', '.join(short)}")
        if ok:
            for name in EXPECTED:
                print(f"  {name}: {(TARGET/name).stat().st_size/1e6:.0f} MB")
            print("  OK")
        elif not verbose:
            pass
        else:
            print(f"  not complete under {TARGET.relative_to(ROOT)}")
    return ok


def progress(done, total):
    """One-line download progress bar."""
    if not total:
        sys.stdout.write(f"\r  {done/1e6:.0f} MB")
    else:
        pct = done / total * 100
        sys.stdout.write(f"\r  [{'#'*int(pct/2.5):<40}] {pct:5.1f}%  "
                         f"{done/1e6:.0f}/{total/1e6:.0f} MB")
    sys.stdout.flush()


def download():
    """Download the archive, extract it, and drop the packaging leftovers."""
    TARGET.mkdir(parents=True, exist_ok=True)
    zpath = TARGET / 'car-hacking.zip'
    print("HCRL Car-Hacking dataset (Korea University)")
    print("  cite Song et al. 2020 and Seo et al. 2018 -- see this file's header")
    done = 0
    with urllib.request.urlopen(URL, timeout=180) as r, open(zpath, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            progress(done, total)
    print()
    if zpath.stat().st_size < 900_000_000:
        zpath.unlink()
        raise SystemExit(
            f"archive is only {done/1e6:.0f} MB, expected ~953 MB. A truncated "
            "archive would extract into short captures and a wrong result, so "
            "the partial file was removed.")
    print("  extracting...")
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if m in ('/',) or m.endswith('.7z'):
                continue
            z.extract(m, TARGET)
    zpath.unlink()
    print(f"  extracted to {TARGET.relative_to(ROOT)}")


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
