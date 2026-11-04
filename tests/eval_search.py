#!/usr/bin/env python3
"""Accuracy harness for video search.

Runs a set of labeled queries against a live API and reports, per query, where
the expected video landed. This is the measuring stick so search changes are
judged by numbers, not by eyeballing three queries.

NOTE: this runner is GENERIC, but the shipped label files are EXAMPLES tied to
one specific video library — their video names will not exist in yours. To use
the harness on your own library, point --labels at a JSON file of your own
queries + expected video names (same format, see below).

Label sets:
    dev   tests/search_labels.json       queries the heuristics were tuned on
    test  tests/search_labels_test.json  held-out queries written WITHOUT looking
                                         at the heuristics; never tune against it

Usage:
    python tests/eval_search.py                          # dev set, localhost:8000
    python tests/eval_search.py --set test               # held-out set
    python tests/eval_search.py --set all                # dev + test
    python tests/eval_search.py --base http://host:8000
    python tests/eval_search.py --labels my_labels.json  # your own library
    python tests/eval_search.py --disable anchor,llm     # run with heuristics off
    python tests/eval_search.py --ablate                 # baseline + each heuristic off

Ablation turns each named search heuristic off one at a time (via the /search
`disable` param) and reports pass rate, mean expected-rank and noise per run, so
a heuristic whose removal doesn't hurt (especially on the held-out set) can be
deleted.

Each case in the labels file:
    {"q": "...", "expect_top": "name"}     # must be the #1 result
    {"q": "...", "expect_in": ["a","b"]}   # all listed must appear in results
    optional "limit": N, optional "sources": "both"|"video"|"audio"
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
LABEL_SETS = {
    "dev": [TESTS_DIR / "search_labels.json"],
    "test": [TESTS_DIR / "search_labels_test.json"],
}
LABEL_SETS["all"] = LABEL_SETS["dev"] + LABEL_SETS["test"]

# Must match SEARCH_HEURISTICS in src/api/server.py.
HEURISTICS = ["boost", "broad_keyword", "distinctive", "degenerate",
              "templated", "short_damp", "anchor", "llm"]


def search(base, q, limit, sources="both", disable=""):
    params = {"q": q, "limit": limit, "sources": sources}
    if disable:
        params["disable"] = disable
    url = f"{base}/search?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=180) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from {url}: {e.read().decode(errors='replace')}")
    # A label may name a video by display name, file stem or video id; the API
    # reports the display name, so resolve each result to whichever one matches.
    return [
        {x.get("name") or "", Path(x.get("path") or "").stem, x.get("video_id") or ""}
        for x in data.get("results", [])
    ]


def rank_of(label, results):
    return next((i + 1 for i, ids in enumerate(results) if label in ids), None)


def load_cases(paths):
    cases = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"(skipping missing labels file {p})", file=sys.stderr)
            continue
        cases.extend(json.load(open(p)).get("cases", []))
    return cases


def run(base, cases, default_limit, disable="", verbose=True):
    passed = total_noise = rank_sum = rank_n = 0
    failed = []
    t0 = time.time()
    for c in cases:
        q = c["q"]
        results = search(base, q, c.get("limit", default_limit), c.get("sources", "both"), disable)
        names = [sorted(ids - {""}, key=len)[0] if ids - {""} else "?" for ids in results]
        ok = True
        detail = ""
        wanted = []
        if "expect_top" in c:
            wanted = [c["expect_top"]]
            top = names[0] if names else "(none)"
            ok = rank_of(c["expect_top"], results) == 1
            detail = f"top={top} want={c['expect_top']}"
        elif "expect_in" in c:
            wanted = c["expect_in"]
            ok = all(rank_of(v, results) for v in wanted)
            ranks = ", ".join(f"{v}@{rank_of(v, results)}" if rank_of(v, results) else f"{v}=MISS" for v in wanted)
            detail = f"{ranks}; noise={len(names)-len(wanted)}"
        total_noise += max(0, len(names) - len(wanted))
        for v in wanted:
            r = rank_of(v, results)
            if r:
                rank_sum += r
                rank_n += 1
        passed += ok
        if not ok:
            failed.append(q)
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {q!r}  {detail}  results={names[:8]}")
    return {
        "passed": passed,
        "total": len(cases),
        "mean_rank": rank_sum / rank_n if rank_n else 0.0,
        "found": rank_n,
        "noise": total_noise,
        "failed": failed,
        "secs": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--set", choices=sorted(LABEL_SETS), default="dev",
                    help="which built-in label set to run (ignored if --labels is given)")
    ap.add_argument("--labels", nargs="+", help="explicit labels file(s); overrides --set")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--disable", default="", help="comma-separated heuristics to turn off")
    ap.add_argument("--ablate", action="store_true",
                    help="run baseline, then once per heuristic with it disabled")
    args = ap.parse_args()

    cases = load_cases(args.labels or LABEL_SETS[args.set])
    if not cases:
        sys.exit("No eval cases found — add queries to the labels file(s).")

    if not args.ablate:
        s = run(args.base, cases, args.limit, args.disable)
        print(f"\n{s['passed']}/{s['total']} passed | mean expected-rank {s['mean_rank']:.2f} "
              f"| total noise {s['noise']}")
        sys.exit(0 if s["passed"] == s["total"] else 1)

    extra = [h.strip() for h in args.disable.split(",") if h.strip()]
    configs = [("baseline", extra)] + [(f"-{h}", extra + [h]) for h in HEURISTICS if h not in extra]
    rows = []
    for label, off in configs:
        print(f"\n=== {label} (disable={','.join(off) or 'none'}) ===")
        s = run(args.base, cases, args.limit, ",".join(off))
        rows.append((label, s))

    base = rows[0][1]
    print(f"\nAblation over {len(cases)} cases ({args.labels or args.set}):\n")
    print(f"{'config':<16}{'pass':>7}{'Δpass':>7}{'found':>7}{'mean rank':>11}{'noise':>7}{'Δnoise':>8}{'secs':>7}  newly failing")
    for label, s in rows:
        newly = [q for q in s["failed"] if q not in base["failed"]]
        fixed = [q for q in base["failed"] if q not in s["failed"]]
        note = "; ".join([*(repr(q) for q in newly), *(f"+fixed {q!r}" for q in fixed)])
        print(f"{label:<16}{s['passed']:>4}/{s['total']:<2}{s['passed']-base['passed']:>+7}{s['found']:>7}"
              f"{s['mean_rank']:>11.2f}{s['noise']:>7}{s['noise']-base['noise']:>+8}{s['secs']:>7.1f}  {note}")
    print("\nA heuristic whose removal leaves pass/rank/noise unchanged (or better) is a removal candidate.")


if __name__ == "__main__":
    main()
