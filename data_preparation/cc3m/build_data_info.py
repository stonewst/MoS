import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

META_FIELDS = ["url", "key", "status", "error_message", "width", "height",
               "original_width", "original_height", "exif", "md5"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build data_info.json from img2dataset outputs.")
    parser.add_argument("--img2dataset_dir", type=str, required=True,
                        help="img2dataset output folder "
                             "(contains {shard}.parquet and {shard}/ image dirs).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save data_info.json.")
    parser.add_argument("--caption_col", type=str, default="caption")
    parser.add_argument("--image_ext", type=str, default="jpg")
    parser.add_argument("--keep_failed", action="store_true",
                        help="Also keep pairs whose download did not succeed.")
    return parser.parse_args()


def to_native(v):
    """Convert numpy/bytes/NaN values into JSON-serializable Python natives."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        v = float(v)
        return None if math.isnan(v) else v
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def main():
    args = parse_args()
    parquet_files = sorted(glob.glob(os.path.join(args.img2dataset_dir, "*.parquet")))
    assert parquet_files, "No .parquet found in {}".format(args.img2dataset_dir)

    data_info, n_total, n_ok = [], 0, 0
    for pq in parquet_files:
        shard = os.path.splitext(os.path.basename(pq))[0]   # e.g. "00000"
        df = pd.read_parquet(pq)
        for row in df.to_dict("records"):
            n_total += 1
            success = str(row.get("status", "")) == "success"
            if not success and not args.keep_failed:
                continue
            n_ok += int(success)
            key = str(row["key"])
            caption = to_native(row.get(args.caption_col, row.get("caption", "")))
            data = {
                "filename": os.path.join(shard, "{}.{}".format(key, args.image_ext)),
                "caption": caption if caption is not None else "",
            }
            for field in META_FIELDS:
                if field in row:
                    data[field] = to_native(row[field])
            data_info.append(data)

    print("Scanned {} rows across {} shards; {} succeeded, {} kept.".format(
        n_total, len(parquet_files), n_ok, len(data_info)))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(data_info, f, ensure_ascii=False)
    print("Saved data_info to {}".format(args.output_path))


if __name__ == "__main__":
    main()
