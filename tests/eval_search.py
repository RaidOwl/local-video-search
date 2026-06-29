#!/usr/bin/env python3
"""Accuracy harness for video search.

Runs a set of labeled queries against a live API and reports, per query, where
the expected video landed. This is the measuring stick so search changes are
judged by numbers, not by eyeballing three queries.

Usage:
    python tests/eval_search.py                       # localhost:8000
    python tests/eval_search.py --base http://host:8000
    python tests/eval_search.py --labels tests/search_labels.json

Each case in the labels file:
    {"q": "...", "expect_top": "name"}     # must be the #1 result
    {"q": "...", "expect_in": ["a","b"]}   # all listed must appear in results
    optional "enhanced": true/false, optional "limit": N
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request


def search(base, q, enhanced, limit):
    params = urllib.parse.urlencode({"q": q, "enhanced": str(enhanced).lower(), "limit": limit})
    with urllib.request.urlopen(f"{base}/search?{params}", timeout=120) as r:
        data = json.load(r)
    return [x.get("name", x.get("id", "")) for x in data.get("results", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--labels", default="tests/search_labels.json")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    cases = json.load(open(args.labels))["cases"]
    passed = 0
    total_noise = 0
    rank_sum = 0
    rank_n = 0
    for c in cases:
        q = c["q"]
        names = search(args.base, q, c.get("enhanced", True), c.get("limit", args.limit))
        ok = True
        detail = ""
        wanted = []
        if "expect_top" in c:
            wanted = [c["expect_top"]]
            top = names[0] if names else "(none)"
            ok = top == c["expect_top"]
            detail = f"top={top} want={c['expect_top']}"
        elif "expect_in" in c:
            wanted = c["expect_in"]
            missing = [v for v in c["expect_in"] if v not in names]
            ok = not missing
            ranks = ", ".join(f"{v}@{names.index(v)+1}" if v in names else f"{v}=MISS" for v in c["expect_in"])
            detail = f"{ranks}; noise={len(names)-len(c['expect_in'])}"
        total_noise += max(0, len(names) - len(wanted))
        for v in wanted:
            if v in names:
                rank_sum += names.index(v) + 1
                rank_n += 1
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {q!r}  {detail}  results={names[:8]}")
    mean_rank = rank_sum / rank_n if rank_n else 0
    print(f"\n{passed}/{len(cases)} passed | mean expected-rank {mean_rank:.2f} | total noise {total_noise}")
    sys.exit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
