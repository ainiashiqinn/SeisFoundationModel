import torch


def random_masking(x: torch.Tensor, mask_ratio: float):
    """
    Per-sample random masking by shuffling.

    Args:
        x: (B, N, D) token sequence (no cls token).
        mask_ratio: fraction of tokens to mask (drop from encoder input).

    Returns:
        x_kept:      (B, N_keep, D)  visible tokens only
        mask:        (B, N)          1 = masked, 0 = kept (in original order)
        ids_restore: (B, N)          permutation that restores original order
                                     when applied to [kept | mask_tokens].
        ids_keep:    (B, N_keep)     original indices of the kept tokens
                                     (needed by RoPE so the kept tokens can be
                                     rotated by their original positions).
    """
    B, N, D = x.shape
    len_keep = max(1, int(N * (1.0 - mask_ratio)))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_kept, mask, ids_restore, ids_keep
