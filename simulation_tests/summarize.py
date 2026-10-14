"""Summarize a replay report by scenario group."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    report = json.loads((args.dataset / "report.json").read_text())
    groups_by_name = {f["name"]: f.get("group", "baseline") for f in manifest["frames"]}
    frames_by_name = {f["name"]: f for f in manifest["frames"]}
    groups = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "missing_or_invalid": 0,
                                  "unexpected_ids": 0, "position_failed": 0})
    failures = []
    for frame in report["frames"]:
        group = groups[groups_by_name[frame["name"]]]
        group["total"] += 1
        group["passed"] += int(frame["passed"])
        group["failed"] += int(not frame["passed"])
        group["missing_or_invalid"] += int(bool(frame["missing_or_invalid_ids"]))
        group["unexpected_ids"] += int(bool(frame.get("unexpected_ids", [])))
        group["position_failed"] += int(not frame.get("position_passed", True))
        if not frame["passed"]:
            fixture = frames_by_name[frame["name"]]
            failures.append({"name": frame["name"], "image": fixture["image"],
                             "case": fixture["case"], "ground_truth": fixture["camera_position"],
                             "result": frame})
    summary = {"total": len(report["frames"]), "passed": sum(g["passed"] for g in groups.values()),
               "failed": sum(g["failed"] for g in groups.values()), "groups": dict(groups)}
    (args.dataset / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.dataset / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"Replay: {summary['passed']}/{summary['total']} pass; {summary['failed']} fail")
    for name, group in groups.items():
        print(f"  {name}: {group['passed']}/{group['total']} pass, "
              f"{group['missing_or_invalid']} missing/invalid, "
              f"{group['unexpected_ids']} unexpected IDs, {group['position_failed']} position failures")


if __name__ == "__main__":
    main()
