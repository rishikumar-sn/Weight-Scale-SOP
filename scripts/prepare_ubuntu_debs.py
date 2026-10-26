"""Online builder: collect Ubuntu amd64 runtime .debs from official package indexes.

Downloads the dependency closure (Depends/Pre-Depends, not Recommends).
The target uses apt --no-download, which checks actual installed-system constraints.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import cmp_to_key
import hashlib
import json
import lzma
from pathlib import Path
import re
import urllib.request

from prepare_offline_bundle import fetch


def order(c):
    if c == "~": return -1
    if not c: return 0
    return ord(c) if c.isalpha() else ord(c) + 256


def part_compare(a, b):
    while a or b:
        while (a and not a[0].isdigit()) or (b and not b[0].isdigit()):
            x, y = a[:1], b[:1]
            if order(x) != order(y): return 1 if order(x) > order(y) else -1
            a, b = a[bool(x):], b[bool(y):]
        a, b = a.lstrip("0"), b.lstrip("0")
        x = re.match(r"\d*", a)[0]
        y = re.match(r"\d*", b)[0]
        if len(x) != len(y): return 1 if len(x) > len(y) else -1
        if x != y: return 1 if x > y else -1
        a, b = a[len(x):], b[len(y):]
    return 0


def compare(a, b):
    def split(v):
        epoch, sep, rest = v.partition(":")
        epoch, v = (int(epoch), rest) if sep else (0, v)
        upstream, sep, revision = v.rpartition("-")
        return epoch, upstream if sep else v, revision if sep else "0"
    a, b = split(a), split(b)
    return (a[0] > b[0]) - (a[0] < b[0]) or part_compare(a[1], b[1]) or part_compare(a[2], b[2])


def index(cache, suite, component):
    path = cache / f"{suite}-{component}.xz"
    fetch(f"https://archive.ubuntu.com/ubuntu/dists/{suite}/{component}/binary-amd64/Packages.xz", path)
    with lzma.open(path, "rt", encoding="utf-8") as source:
        record = {}
        for line in source:
            if line == "\n":
                if record:
                    yield record
                record = {}
            elif not line[0].isspace():
                key, _, value = line.rstrip("\n").partition(": ")
                if key in {"Package", "Version", "Filename", "SHA256", "Depends", "Pre-Depends", "Provides", "Architecture"}:
                    record[key] = value
        if record: yield record


def build(destination, codename, version):
    cache = Path(__file__).resolve().parents[1] / "offline/build-logs/apt-indexes"
    cache.mkdir(parents=True, exist_ok=True)
    records = {}
    for suite in [codename, codename + "-updates", codename + "-security"]:
        for component in ["main", "universe"]:
            print("Reading", suite, component, flush=True)
            for record in index(cache, suite, component):
                name = record["Package"]
                if name not in records or compare(record["Version"], records[name]["Version"]) > 0:
                    records[name] = record
    providers = {}
    for record in records.values():
        for value in record.get("Provides", "").split(","):
            name = re.split(r"[\s(]", value.strip())[0]
            if name:
                providers.setdefault(name, []).append(record["Package"])
    roots = ["v4l-utils", "util-linux", "libgl1", "libgomp1", "fonts-dejavu-core", "libxkbcommon-x11-0",
             "libxcb-xinerama0", "libxcb-cursor0", "libxcb-icccm4", "libxcb-image0", "libxcb-keysyms1",
             "libxcb-render-util0", "libxcb-xfixes0", "libxcb-shape0", "libxcb-randr0", "libxcb-sync1",
             "libxcb-xkb1", "libxrender1", "libxi6", "libsm6", "libice6", "libdbus-1-3", "libsqlite3-0",
             "libbz2-1.0", "liblzma5", "zlib1g", "libncursesw6", "libffi8", "libfontconfig1",
             "libfreetype6", "libegl1", "libxcb-xinput0"]
    roots += ["libglib2.0-0", "libssl3", "libreadline8"] if codename == "jammy" else ["libglib2.0-0t64", "libssl3t64", "libreadline8t64"]
    selected, pending = {}, list(roots)
    while pending:
        name = pending.pop()
        if name in selected: continue
        record = records[name]
        selected[name] = record
        for group in (record.get("Pre-Depends", "") + "," + record.get("Depends", "")).split(","):
            if not group.strip(): continue
            alternatives = [re.split(r"[\s(:\[]", p.strip())[0] for p in group.split("|")]
            choice = next((n for n in alternatives if n in records), None)
            if choice is None:
                choice = next((providers[n][0] for n in alternatives if n in providers), None)
            if choice is None: raise RuntimeError(f"Unresolved dependency: {name}: {group}")
            pending.append(choice)
    folder = destination / f"offline/ubuntu-{version}-amd64"
    folder.mkdir(parents=True, exist_ok=True)

    def download(record):
        url = "https://archive.ubuntu.com/ubuntu/" + record["Filename"]
        path = folder / Path(record["Filename"]).name
        fetch(url, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["SHA256"]:
            raise RuntimeError(f"Ubuntu package hash mismatch: {path}")
        print(f"Ubuntu {version}: {record['Package']} OK", flush=True)
        return {"name": record["Package"], "version": record["Version"], "file": path.name, "sha256": record["SHA256"], "url": url}

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(download, selected.values()))
    (folder / "packages.json").write_text(json.dumps({"ubuntu": version, "roots": roots, "packages": manifest}, indent=2), encoding="utf-8")
    print(f"Ubuntu {version}: {len(manifest)} packages downloaded and hash checked", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    for codename, version in [("jammy", "22.04"), ("noble", "24.04")]:
        build(args.destination, codename, version)
