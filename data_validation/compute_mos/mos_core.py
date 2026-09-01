import torch


def mixture_of_scores(s, temp_min=0.5, temp_max=1.5, eps=1e-8):
    M = s.shape[1]
    # --- (1) density: negative average L1 distance to the other scores, Eq. (4) ---
    density = -(s[:, :, None] - s[:, None, :]).abs().sum(-1) / (M - 1)   # [N, M]
    # --- (2) uncertainty-aware temperature from per-sample std, Eq. (6) ---
    std = s.std(-1, unbiased=False)                                      # [N]
    temp = (temp_max - temp_min) / (std.max() - std.min() + eps) * std \
        + (temp_min * std.max() - temp_max * std.min()) / (std.max() - std.min() + eps)
    # --- (3) softmax-weighted ensemble, Eq. (5) & (7) ---
    mos = (torch.softmax(density / temp[:, None], dim=-1) * s).sum(-1)   # [N]
    return mos


def normalize_scores(s, mode="minmax", eps=1e-8):
    if mode == "none":
        return s
    if mode == "minmax":
        lo = s.min(dim=0, keepdim=True).values
        hi = s.max(dim=0, keepdim=True).values
        return (s - lo) / (hi - lo + eps)
    if mode == "zscore":
        mean = s.mean(dim=0, keepdim=True)
        std = s.std(dim=0, keepdim=True)
        return (s - mean) / (std + eps)
    raise ValueError("Unknown normalize mode: {}".format(mode))


if __name__ == "__main__":
    # A minimal self-check on random data.
    torch.manual_seed(0)
    scores = torch.rand(5, 4)
    print("baseline scores:\n", scores)
    print("MoS:\n", mixture_of_scores(normalize_scores(scores)))
