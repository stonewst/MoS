import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Filter image-text data by quality score.")
    parser.add_argument("--data_info_path", type=str, required=True,
                        help="Input data_info .json; each dict must carry --score_key.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save the kept (filtered) data_info .json.")
    parser.add_argument("--score_key", type=str, default="mos_score",
                        help="Field to sort by (default: mos_score).")
    parser.add_argument("--keep_ratio", type=float, default=0.8,
                        help="Fraction of highest-scoring data to keep, in (0, 1]. "
                             "e.g. 0.8 keeps the top 80%% and removes the lowest 20%%.")
    parser.add_argument("--ascending", action="store_true",
                        help="Keep the LOWEST-scoring data instead of the highest.")
    parser.add_argument("--removed_path", type=str, default=None,
                        help="Optional: also save the removed data_info .json.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 < args.keep_ratio <= 1.0):
        raise ValueError("--keep_ratio must be in (0, 1], got {}".format(args.keep_ratio))

    print("\nLoading data_info: {}".format(args.data_info_path))
    with open(args.data_info_path, "r") as f:
        data_info = json.load(f)
    print("{} data loaded.".format(len(data_info)))

    # keep only entries that actually carry the score
    scored = [d for d in data_info if args.score_key in d and d[args.score_key] is not None]
    dropped = len(data_info) - len(scored)
    if dropped:
        print("[warn] {} data have no '{}' and are dropped.".format(dropped, args.score_key))
    if not scored:
        raise ValueError("No data carry the score key '{}'.".format(args.score_key))

    # sort by score (highest quality first by default)
    scored.sort(key=lambda d: float(d[args.score_key]), reverse=not args.ascending)

    n_keep = int(round(len(scored) * args.keep_ratio))
    n_keep = max(1, min(n_keep, len(scored)))
    kept, removed = scored[:n_keep], scored[n_keep:]

    print("Sorting by '{}' ({}); keeping {}/{} = {:.1f}%.".format(
        args.score_key, "ascending" if args.ascending else "descending",
        n_keep, len(scored), 100.0 * n_keep / len(scored)))

    print("Saving kept data to {}".format(args.output_path))
    with open(args.output_path, "w") as f:
        json.dump(kept, f, ensure_ascii=False)

    if args.removed_path is not None:
        print("Saving removed data to {}".format(args.removed_path))
        with open(args.removed_path, "w") as f:
            json.dump(removed, f, ensure_ascii=False)

    print("Done. kept={}, removed={}.".format(len(kept), len(removed)))


if __name__ == "__main__":
    main()
