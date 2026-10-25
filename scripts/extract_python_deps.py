#!/usr/bin/env python3
"""Extract Python dependency specs from service Dockerfiles.

The Dockerfiles are the source of truth for what gets installed. This script
turns their `pip install` lines into a requirements file that pip-audit (and
any other scanner) can consume, so there is no second list to keep in sync.

Usage:
    python scripts/extract_python_deps.py                 # all Dockerfiles
    python scripts/extract_python_deps.py --dockerfile laya/Dockerfile
"""
import argparse
import glob
import re
import sys

FLAGS_WITH_VALUE = {"--index-url", "--extra-index-url", "--find-links", "-i", "-f"}


def pip_install_specs(text: str) -> list[str]:
    # Join backslash-continued lines first so multi-line installs parse whole.
    lines = text.splitlines()
    joined: list[str] = []
    buf = ""
    for line in lines:
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
        else:
            joined.append(buf + line)
            buf = ""
    if buf:
        joined.append(buf)

    specs: list[str] = []
    for line in joined:
        m = re.search(r"\bpip install\b(.*)", line)
        if not m:
            continue
        tokens = m.group(1).split()
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in FLAGS_WITH_VALUE:
                i += 2
                continue
            if tok.startswith(("-", "http://", "https://")):
                i += 1
                continue
            specs.append(tok)
            i += 1
    return specs


def package_name(spec: str) -> str:
    return re.split(r"[<>=!~]", spec, maxsplit=1)[0].lower().replace("-", "_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dockerfile", help="scan a single Dockerfile instead of all")
    args = ap.parse_args()
    paths = [args.dockerfile] if args.dockerfile else sorted(glob.glob("**/Dockerfile", recursive=True))
    specs: list[str] = []
    for path in paths:
        specs.extend(pip_install_specs(open(path).read()))
    # Deduplicate by package name (keep first occurrence) so pip-audit can resolve.
    seen: set[str] = set()
    out: list[str] = []
    for s in specs:
        name = package_name(s)
        if name not in seen:
            seen.add(name)
            out.append(s)
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EOF