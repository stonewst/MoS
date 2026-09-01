import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser(description="Compute BLIP2 baseline quality score.")
    # data
    parser.add_argument("--data_info_path", type=str, required=True)
    parser.add_argument("--image_root_path", type=str, required=True)
    parser.add_argument("--start_id", type=int, default=-1)
    parser.add_argument("--end_id", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    # model (passed to lavis.models.load_model_and_preprocess)
    parser.add_argument("--name", type=str, default="blip2_image_text_matching",
                        help="LAVIS model name for BLIP2 ITM.")
    parser.add_argument("--model_type", type=str, default="pretrain",
                        help="LAVIS model_type, e.g. pretrain (ViT-g/14) or coco.")
    # save
    parser.add_argument("--save_root_path", type=str, required=True)
    parser.add_argument("--save_filename", type=str, default="data_info_blip2.json")
    parser.add_argument("--save_key", type=str, default="blip2",
                        help="Prefix for the two stored keys: "
                             "`<save_key>_itm-score` and `<save_key>_itc-similarity`.")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.save_root_path, exist_ok=True)
    save_path = os.path.join(args.save_root_path, args.save_filename)

    # --- load dataset ---
    print("\nLoading data_info: {}".format(args.data_info_path))
    with open(args.data_info_path, "r") as f:
        dataset = json.load(f)
    if args.start_id != -1 and args.end_id != -1:
        dataset = dataset[args.start_id:args.end_id]
        print("Selected a subset from {} to {}".format(args.start_id, args.end_id))
    num_data = len(dataset)
    print("{} image-text pairs to be processed.".format(num_data))

    # --- load BLIP2 model via LAVIS ---
    print("\nLoading BLIP2 model: {} ({})".format(args.name, args.model_type))
    from lavis.models import load_model_and_preprocess
    model, vis_processors, text_processors = load_model_and_preprocess(
        name=args.name, model_type=args.model_type, is_eval=True, device=args.device)
    model = model.eval()

    itm_key = "{}_itm-score".format(args.save_key)
    itc_key = "{}_itc-similarity".format(args.save_key)

    # --- score each pair (BLIP2 ITM/ITC is computed per single pair) ---
    print("\nStart scoring ...")
    new_data_info = []
    for data in tqdm(dataset):
        image_path = os.path.join(args.image_root_path, data["filename"])
        raw_image = Image.open(image_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(args.device)
        caption = text_processors["eval"](data["caption"])

        itm_output = model({"image": image, "text_input": caption}, match_head="itm")
        itm_score = torch.softmax(itm_output, dim=1)[:, 1]              # prob of match
        itc_output = model({"image": image, "text_input": caption}, match_head="itc")

        data[itm_key] = float(itm_score[0].detach().cpu())
        # ITC may return a per-query-token similarity vector; take the max.
        data[itc_key] = float(itc_output.detach().cpu().max())
        new_data_info.append(data)

    assert len(new_data_info) == num_data
    print("\nSaving to {}".format(save_path))
    with open(save_path, "w") as f:
        json.dump(new_data_info, f, ensure_ascii=False)
    print("Saved!")


if __name__ == "__main__":
    main()
