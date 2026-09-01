import argparse
import glob
import json


META_KEYS = {"filename", "caption", "url", "key", "status", "error_message",
             "width", "height", "original_width", "original_height",
             "exif", "md5", "org_filename"}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False)


def expand_inputs(inputs):
    paths = []
    for pattern in inputs:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    seen, ordered = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def concat(args):
    all_data = []
    for path in expand_inputs(args.inputs):
        print("Reading {}".format(path))
        all_data += load_json(path)
    if args.check_unique:
        filenames = [d["filename"] for d in all_data]
        assert len(filenames) == len(set(filenames)), "Duplicated filenames found!"
    print("Merged {} data. Saving to {}".format(len(all_data), args.output))
    save_json(all_data, args.output)


def join(args):
    paths = expand_inputs(args.inputs)
    base = None
    for path in paths:
        print("Reading {}".format(path))
        data_list = load_json(path)
        index = {d["filename"]: d for d in data_list}
        if base is None:
            # first file defines the set of samples and their meta info
            base = {fn: dict(d) for fn, d in index.items()}
            continue
        for fn, d in index.items():
            if fn not in base:
                continue
            for k, v in d.items():
                if k not in META_KEYS:
                    base[fn][k] = v

    merged = list(base.values())
    score_keys = sorted({k for d in merged for k in d if k not in META_KEYS})
    print("Merged {} data with {} baseline score keys:".format(len(merged), len(score_keys)))
    for k in score_keys:
        print("  - {}".format(k))
    save_json(merged, args.output)
    print("Saved to {}".format(args.output))


def parse_args():
    parser = argparse.ArgumentParser(description="Merge baseline-score files.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_concat = sub.add_parser("concat", help="Concatenate splits of one score.")
    p_concat.add_argument("--inputs", nargs="+", required=True)
    p_concat.add_argument("--output", type=str, required=True)
    p_concat.add_argument("--check_unique", action="store_true")
    p_concat.set_defaults(func=concat)

    p_join = sub.add_parser("join", help="Join different scores by filename.")
    p_join.add_argument("--inputs", nargs="+", required=True)
    p_join.add_argument("--output", type=str, required=True)
    p_join.set_defaults(func=join)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
