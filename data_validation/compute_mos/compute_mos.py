import argparse
import json

import torch

from mos_core import mixture_of_scores, normalize_scores

META_KEYS = {"filename", "caption", "url", "key", "status", "error_message",
             "width", "height", "original_width", "original_height",
             "exif", "md5", "org_filename"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute the MoS ensemble score.")
    parser.add_argument("--data_info_path", type=str, required=True,
                        help="A data_info .json where each dict holds several baseline scores.")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--score_keys", nargs="+", default=None,
                        help="Which baseline-score keys to ensemble.")
    parser.add_argument("--auto_keys", action="store_true",
                        help="Auto-detect baseline-score keys (all non-meta keys).")
    parser.add_argument("--save_key", type=str, default="mos_score")
    parser.add_argument("--normalize", type=str, default="minmax",
                        choices=["minmax", "zscore", "none"],
                        help="Per-score normalization before ensembling.")
    parser.add_argument("--temp_min", type=float, default=0.5)
    parser.add_argument("--temp_max", type=float, default=1.5)
    return parser.parse_args()


def resolve_score_keys(data_info, args):
    if args.score_keys:
        keys = args.score_keys
    elif args.auto_keys:
        keys = sorted({k for d in data_info for k in d if k not in META_KEYS})
    else:
        raise ValueError("Please provide --score_keys or set --auto_keys.")
    # keep only keys present in every data dict
    valid = [k for k in keys if all(k in d for d in data_info)]
    missing = set(keys) - set(valid)
    if missing:
        print("[warn] dropping keys not present in all data: {}".format(sorted(missing)))
    if len(valid) < 2:
        raise ValueError("MoS needs at least 2 baseline scores, got {}.".format(valid))
    return valid


def main():
    args = parse_args()

    print("\nLoading data_info: {}".format(args.data_info_path))
    with open(args.data_info_path, "r") as f:
        data_info = json.load(f)
    print("{} data loaded.".format(len(data_info)))

    score_keys = resolve_score_keys(data_info, args)
    print("Ensembling {} baseline scores:".format(len(score_keys)))
    for k in score_keys:
        print("  - {}".format(k))

    # build the [N, M] score matrix
    s = torch.tensor(
        [[float(d[k]) for k in score_keys] for d in data_info],
        dtype=torch.float32,
    )
    s = normalize_scores(s, mode=args.normalize)

    # the 3-line MoS
    mos = mixture_of_scores(s, temp_min=args.temp_min, temp_max=args.temp_max)

    for d, v in zip(data_info, mos.tolist()):
        d[args.save_key] = float(v)

    print("\nSaving to {}".format(args.output_path))
    with open(args.output_path, "w") as f:
        json.dump(data_info, f, ensure_ascii=False)
    print("Saved! MoS score stored under key '{}'.".format(args.save_key))


if __name__ == "__main__":
    main()
