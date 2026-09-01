import argparse
import json
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser(description="Compute BLIP baseline quality score.")
    # data
    parser.add_argument("--data_info_path", type=str, required=True)
    parser.add_argument("--image_root_path", type=str, required=True)
    parser.add_argument("--start_id", type=int, default=-1)
    parser.add_argument("--end_id", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    # model
    parser.add_argument("--blip_code_path", type=str, required=True,
                        help="Path to the official BLIP repo (for `import models.blip_itm`).")
    parser.add_argument("--blip_model_ckpt", type=str, required=True,
                        help="Path to the BLIP checkpoint (.pth).")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--vit_arch", type=str, default="base", choices=["base", "large"])
    # save
    parser.add_argument("--save_root_path", type=str, required=True)
    parser.add_argument("--save_filename", type=str, default="data_info_blip.json")
    parser.add_argument("--save_key", type=str, default="blip",
                        help="Prefix for the two stored keys: "
                             "`<save_key>_itm-score` and `<save_key>_itc-similarity`.")
    return parser.parse_args()


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])


@torch.no_grad()
def main():
    args = parse_args()
    sys.path.append(args.blip_code_path)
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

    # --- load BLIP model ---
    print("\nLoading BLIP model from ckpt: {}".format(args.blip_model_ckpt))
    from models.blip_itm import blip_itm
    model = blip_itm(
        pretrained=args.blip_model_ckpt,
        image_size=args.image_size,
        vit=args.vit_arch,
        med_config=os.path.join(args.blip_code_path, "configs/med_config.json"),
    )
    model = model.eval().to(device=args.device)

    transform = build_transform(args.image_size)
    itm_key = "{}_itm-score".format(args.save_key)
    itc_key = "{}_itc-similarity".format(args.save_key)

    # --- score each pair (BLIP ITM/ITC is computed per single pair) ---
    print("\nStart scoring ...")
    new_data_info = []
    for data in tqdm(dataset):
        image_path = os.path.join(args.image_root_path, data["filename"])
        image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(args.device)
        caption = data["caption"]

        itm_output = model(image, caption, match_head="itm")            # (1, 2)
        itm_score = torch.softmax(itm_output, dim=1)[:, 1]              # (1,)
        itc_similarity = model(image, caption, match_head="itc")        # (1, 1)

        data[itm_key] = float(itm_score[0].detach().cpu())
        data[itc_key] = float(itc_similarity[0][0].detach().cpu())
        new_data_info.append(data)

    assert len(new_data_info) == num_data
    print("\nSaving to {}".format(save_path))
    with open(save_path, "w") as f:
        json.dump(new_data_info, f, ensure_ascii=False)
    print("Saved!")


if __name__ == "__main__":
    main()
