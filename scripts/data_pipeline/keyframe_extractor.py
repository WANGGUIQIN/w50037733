"""Extract keyframes based on action-change magnitude."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def extract_keyframes(
    actions: np.ndarray,
    num_keyframes: int = 5,
    min_gap_ratio: float = 1 / 6,
) -> list[int]:
    """Select keyframes at moments of largest action change.

    Args:
        actions: [T, action_dim] array of actions per timestep.
        num_keyframes: total keyframes to return (including first and last).
        min_gap_ratio: minimum gap between keyframes as fraction of T.

    Returns:
        Sorted list of keyframe indices.
    """
    T = len(actions)
    if T <= num_keyframes:
        return list(range(T))

    indices = [0, T - 1]
    num_middle = num_keyframes - 2
    if num_middle <= 0:
        return sorted(indices)

    # Compute action delta magnitudes
    deltas = np.linalg.norm(np.diff(actions, axis=0), axis=-1)  # [T-1]

    # Smooth to avoid noise spikes
    if len(deltas) > 7:
        deltas = gaussian_filter1d(deltas.astype(np.float64), sigma=3)

    # Greedily pick peaks with minimum gap
    min_gap = max(int(T * min_gap_ratio), 1)
    used = set(indices)

    for _ in range(num_middle):
        best_idx, best_val = -1, -1.0
        for i in range(len(deltas)):
            if any(abs(i - u) < min_gap for u in used):
                continue
            if deltas[i] > best_val:
                best_val = deltas[i]
                best_idx = i
        if best_idx >= 0:
            frame_idx = min(best_idx + 1, T - 1)
            indices.append(frame_idx)
            used.add(frame_idx)
        else:
            # Fallback: uniform spacing
            step = T // (num_middle + 1)
            indices.append(step * (len(indices) - 1))

    return sorted(set(indices))[:num_keyframes]


def uniform_keyframes(num_frames: int, num_keyframes: int = 5) -> list[int]:
    """Fallback: uniformly spaced keyframes when no action data."""
    if num_frames <= num_keyframes:
        return list(range(num_frames))
    step = num_frames / num_keyframes
    indices = [int(i * step) for i in range(num_keyframes - 1)] + [num_frames - 1]
    return sorted(set(indices))[:num_keyframes]
