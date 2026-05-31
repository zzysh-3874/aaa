"""Dump selected TensorBoard scalars from an event file to stdout as a table.

Usage:
    python scripts/dump_tb_scalars.py <run_dir> [--every N] [--tags tagA,tagB]

Prints iteration-aligned values for a small set of diagnostic tags so we can
confirm the temporal ordering of a training collapse (e.g. did mean_noise_std
rise BEFORE episode_length fell, or after).
"""
import argparse
import glob
import os

from tensorboard.backend.event_processing import event_accumulator

DEFAULT_TAGS = [
    "Policy/mean_noise_std",
    "Loss/entropy",
    "Loss/surrogate",
    "Loss/value_function",
    "Loss/learning_rate",
    "Train/mean_episode_length",
    "Train/mean_reward",
    "Metrics/base_parkour/terrain_levels",
    "Metrics/base_parkour/how_far_from_start_point",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--tags", type=str, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    args = ap.parse_args()

    tags = args.tags.split(",") if args.tags else DEFAULT_TAGS

    ev_files = glob.glob(os.path.join(args.run_dir, "events.out.tfevents.*"))
    if not ev_files:
        raise SystemExit(f"no event files in {args.run_dir}")
    ev_files.sort(key=os.path.getmtime)

    # Merge all event files (a resumed run may have several).
    series = {t: {} for t in tags}
    for ev in ev_files:
        ea = event_accumulator.EventAccumulator(
            ev, size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        avail = set(ea.Tags().get("scalars", []))
        for t in tags:
            if t in avail:
                for s in ea.Scalars(t):
                    series[t][s.step] = s.value

    # Print which tags were actually found.
    found = [t for t in tags if series[t]]
    missing = [t for t in tags if not series[t]]
    if missing:
        print("# MISSING tags:", ", ".join(missing))
    print("# step\t" + "\t".join(t.split("/")[-1] for t in found))

    all_steps = sorted({st for t in found for st in series[t]})
    for st in all_steps:
        if st < args.start or st > args.end:
            continue
        if st % args.every != 0:
            continue
        row = [str(st)]
        for t in found:
            v = series[t].get(st)
            row.append(f"{v:.4f}" if v is not None else "-")
        print("\t".join(row))


if __name__ == "__main__":
    main()
