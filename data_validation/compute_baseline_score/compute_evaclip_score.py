import argparse
import json
import math
import os

import open_clip
import torch
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # remove the max-pixel limit for large images


def parse_args():
    parser = argparse.ArgumentParser(description="Compute EVA-CLIP baseline quality score.")
    # data
    parser.add_argument("--data_info_path", type=str, required=True,
                        help="Path to the data_info .json file (a list of dicts).")
    parser.add_argument("--image_root_path", type=str, required=True,
                        help="Root directory that stores the images.")
    parser.add_argument("--start_id", type=int, default=-1,
                        help="Start index of the sub-range to process (-1 = from the beginning).")
    parser.add_argument("--end_id", type=int, default=-1,
                        help="End index of the sub-range to process (-1 = to the end).")
    parser.add_argument("--batchsize", type=int, default=64)
    # model
    parser.add_argument("--model_name", type=str, default="EVA02-L-14",
                        help="open_clip model name, e.g. EVA02-L-14 / EVA01-g-14.")
    parser.add_argument("--pretrained", type=str, default="merged2b_s4b_b131k",
                        help="open_clip pretrained tag (see open_clip.list_pretrained()).")
    # save
    parser.add_argument("--save_root_path", type=str, required=True)
    parser.add_argument("--save_filename", type=str, default="data_info_evaclip.json")
    parser.add_argument("--save_key", type=str, default="evaclip_similarity",
                        help="Key under which the score is stored in each data dict.")
    return parser.parse_args()


def build_batches(dataset, batchsize):
    """Split a flat list into a list of batches (each batch is a list of dicts)."""
    num_batch = math.ceil(len(dataset) / batchsize)
    batches = [dataset[i * batchsize:(i + 1) * batchsize] for i in range(num_batch)]
    return batches


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.save_root_path, exist_ok=True)
    save_path = os.path.join(args.save_root_path, args.save_filename)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load dataset ---
    print("\nLoading data_info: {}".format(args.data_info_path))
    with open(args.data_info_path, "r") as f:
        dataset = json.load(f)
    if args.start_id != -1 and args.end_id != -1:
        dataset = dataset[args.start_id:args.end_id]
        print("Selected a subset from {} to {}".format(args.start_id, args.end_id))
    num_data = len(dataset)
    print("{} image-text pairs to be processed.".format(num_data))

    # --- load EVA-CLIP model via open_clip ---
    print("\nLoading EVA-CLIP model: {} ({})".format(args.model_name, args.pretrained))
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.eval()

    # --- compute scores batch by batch ---
    print("\nStart scoring ...")
    new_data_info = []
    for batch in tqdm(build_batches(dataset, args.batchsize)):
        images, texts = [], []
        for data in batch:
            image_path = os.path.join(args.image_root_path, data["filename"])
            images.append(preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0))
            texts.append(tokenizer([data["caption"]]))
        images = torch.cat(images, dim=0).to(device)
        texts = torch.cat(texts, dim=0).to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity per pair
        similarity = (image_features * text_features).sum(dim=-1)
        similarity = similarity.detach().cpu().numpy()

        for idx, data in enumerate(batch):
            data[args.save_key] = float(similarity[idx])
            new_data_info.append(data)

    assert len(new_data_info) == num_data
    print("\nSaving to {}".format(save_path))
    with open(save_path, "w") as f:
        json.dump(new_data_info, f, ensure_ascii=False)
    print("Saved!")


if __name__ == "__main__":
    main()
