#!/usr/bin/env python3
"""Generate a deterministic DUNE postprocess parity fixture.

The C++ unit test reads the generated text file and compares
`DunePostprocessor` against the same numpy logic used by Python `dune.py`.
"""

from __future__ import annotations

from math import floor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "dune_postprocess_parity_fixture.txt"
SEED = 20260711
TEMPORAL_TAG = 1
DIVERSITY_TAG = 2


def quota_counts(total: int, ratios: tuple[float, ...]) -> tuple[int, ...]:
    total = max(0, int(total))
    raw = [min(max(float(ratio), 0.0), 1.0) * float(total) for ratio in ratios]
    counts = [floor(value) for value in raw]
    target = min(total, floor(sum(raw) + 1.0e-9))
    remaining = int(target - sum(counts))
    if remaining > 0:
        remainders = [value - count for value, count in zip(raw, counts)]
        order = sorted(range(len(remainders)), key=lambda idx: (-remainders[idx], idx))
        for idx in order[:remaining]:
            counts[idx] += 1
    return tuple(int(count) for count in counts)


def append_ranked(
    selected: list[int],
    selected_mask: np.ndarray,
    candidates: np.ndarray,
    target_max: int,
    limit: int | None = None,
    candidate_mask: np.ndarray | None = None,
) -> None:
    room = max(0, int(target_max) - len(selected))
    if limit is not None:
        room = min(room, max(0, int(limit)))
    if room <= 0:
        return

    for raw_idx in candidates:
        idx = int(raw_idx)
        if selected_mask[idx]:
            continue
        if candidate_mask is not None and not bool(candidate_mask[idx]):
            continue
        selected_mask[idx] = True
        selected.append(idx)
        room -= 1
        if room <= 0 or len(selected) >= target_max:
            break


def quota_select_indices(
    margin_batch: np.ndarray,
    tags: np.ndarray,
    select_num: int,
    ratios: tuple[float, float, float],
) -> tuple[np.ndarray, dict[str, float]]:
    num_steps, num_points = margin_batch.shape
    k = min(num_points, int(select_num))
    if k <= 0:
        return np.zeros((num_steps, 0), dtype=np.int64), {
            "nearest": 0.0,
            "temporal": 0.0,
            "diversity": 0.0,
        }

    ordered = np.argsort(margin_batch, axis=1, kind="stable")
    nearest_quota, temporal_quota, diversity_quota = quota_counts(k, ratios)
    temporal_mask = tags[0]
    diversity_mask = tags[1]
    temporal_active = temporal_quota > 0 and bool(np.any(temporal_mask))
    diversity_active = diversity_quota > 0 and bool(np.any(diversity_mask))

    selected_indices = np.zeros((num_steps, k), dtype=np.int64)
    nearest_total = 0
    temporal_total = 0
    diversity_total = 0

    for step in range(num_steps):
        if not temporal_active and not diversity_active:
            selected = [int(idx) for idx in ordered[step, :k]]
            selected_indices[step, :] = selected
            nearest_total += min(nearest_quota, k)
            temporal_total += int(np.count_nonzero(temporal_mask[selected]))
            diversity_total += int(np.count_nonzero(diversity_mask[selected]))
            continue

        selected: list[int] = []
        selected_mask = np.zeros((num_points,), dtype=np.bool_)
        append_ranked(selected, selected_mask, ordered[step], k, nearest_quota)
        nearest_total += len(selected)

        if temporal_active:
            already = int(np.count_nonzero(selected_mask & temporal_mask))
            append_ranked(
                selected,
                selected_mask,
                ordered[step],
                k,
                max(0, temporal_quota - already),
                temporal_mask,
            )
        if diversity_active:
            already = int(np.count_nonzero(selected_mask & diversity_mask))
            append_ranked(
                selected,
                selected_mask,
                ordered[step],
                k,
                max(0, diversity_quota - already),
                diversity_mask,
            )
        append_ranked(selected, selected_mask, ordered[step], k)
        selected_indices[step, :] = selected
        temporal_total += int(np.count_nonzero(selected_mask & temporal_mask))
        diversity_total += int(np.count_nonzero(selected_mask & diversity_mask))

    return selected_indices, {
        "nearest": float(nearest_total) / float(num_steps),
        "temporal": float(temporal_total) / float(num_steps),
        "diversity": float(diversity_total) / float(num_steps),
    }


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def write_array(stream, name: str, array: np.ndarray) -> None:
    arr = np.asarray(array)
    if arr.ndim == 1:
        stream.write(f"{name} {arr.shape[0]}\n")
        stream.write(" ".join(f"{float(value):.10g}" for value in arr))
        stream.write("\n")
        return
    if arr.ndim == 2:
        stream.write(f"{name} {arr.shape[0]} {arr.shape[1]}\n")
        for row in range(arr.shape[0]):
            stream.write(" ".join(f"{float(value):.10g}" for value in arr[row]))
            stream.write("\n")
        return
    if arr.ndim == 3:
        stream.write(f"{name} {arr.shape[0]} {arr.shape[1]} {arr.shape[2]}\n")
        for step in range(arr.shape[0]):
            for row in range(arr.shape[1]):
                stream.write(" ".join(f"{float(value):.10g}" for value in arr[step, row]))
                stream.write("\n")
        return
    raise ValueError(f"unsupported fixture array rank: {arr.ndim}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    receding = 3
    steps = receding + 1
    point_dim = 3
    edge_dim = 6
    num_points = 7
    dune_max_num = 9
    select_num = 4
    ratios = (0.5, 0.25, 0.25)

    G = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    h = np.array([0.32, 0.32, 0.32, 0.32, 0.27, 0.27], dtype=np.float32)

    point_flow = rng.uniform(-3.0, 3.0, size=(steps, point_dim, num_points)).astype(np.float32)
    obstacle_points = point_flow + rng.normal(0.0, 0.35, size=point_flow.shape).astype(np.float32)
    raw_mu = rng.normal(0.15, 1.2, size=(steps * num_points, edge_dim)).astype(np.float32)

    rotations = np.stack(
        [
            rotation_matrix(0.05, -0.08, 0.10),
            rotation_matrix(-0.12, 0.04, 0.45),
            rotation_matrix(0.09, 0.11, -0.35),
            rotation_matrix(-0.03, -0.07, 0.85),
        ],
        axis=0,
    )

    tag_mask = np.zeros((2, num_points), dtype=np.bool_)
    tag_mask[0, [1, 4, 6]] = True
    tag_mask[1, [2, 4, 5]] = True
    tag_bits = np.zeros((num_points,), dtype=np.int32)
    tag_bits[tag_mask[0]] |= TEMPORAL_TAG
    tag_bits[tag_mask[1]] |= DIVERSITY_TAG

    projected_mu = np.maximum(raw_mu.astype(np.float32, copy=False), 0.0)
    dual_vector = projected_mu @ G
    dual_norm = np.linalg.norm(dual_vector, axis=1, keepdims=True)
    projected_mu = np.ascontiguousarray(projected_mu / np.maximum(dual_norm, 1.0))
    mu_batch = projected_mu.reshape(steps, num_points, edge_dim).transpose(0, 2, 1)
    lambda_batch = -np.matmul(np.matmul(rotations, G.T), mu_batch)
    margin_batch = np.sum(
        mu_batch * (np.matmul(G[None, :, :], point_flow) - h.reshape(1, edge_dim, 1)),
        axis=1,
    )
    min_margin = float(np.min(margin_batch[0]))
    selected, profile = quota_select_indices(margin_batch, tag_mask, select_num, ratios)
    k = int(selected.shape[1])
    mu_selected = np.take_along_axis(
        mu_batch, np.broadcast_to(selected[:, None, :], (steps, edge_dim, k)), axis=2
    )
    lambda_selected = np.take_along_axis(
        lambda_batch, np.broadcast_to(selected[:, None, :], (steps, point_dim, k)), axis=2
    )
    point_selected = np.take_along_axis(
        obstacle_points, np.broadcast_to(selected[:, None, :], (steps, point_dim, k)), axis=2
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as stream:
        stream.write("DUNE_POSTPROCESS_PARITY_V1\n")
        stream.write(f"seed {SEED}\n")
        stream.write(f"receding {receding}\n")
        stream.write(f"point_dim {point_dim}\n")
        stream.write(f"edge_dim {edge_dim}\n")
        stream.write(f"num_points {num_points}\n")
        stream.write(f"dune_max_num {dune_max_num}\n")
        stream.write(f"select_num {select_num}\n")
        stream.write(f"ratios {ratios[0]:.10g} {ratios[1]:.10g} {ratios[2]:.10g}\n")
        write_array(stream, "G", G)
        write_array(stream, "h", h)
        write_array(stream, "raw_mu", raw_mu)
        write_array(stream, "point_flow", point_flow)
        write_array(stream, "rotations", rotations)
        write_array(stream, "obstacle_points", obstacle_points)
        write_array(stream, "tags", tag_bits)
        stream.write(f"expected_min_margin {min_margin:.10g}\n")
        stream.write(f"expected_selected_count {k}\n")
        stream.write(
            "expected_profile "
            f"{profile['nearest']:.10g} {profile['temporal']:.10g} {profile['diversity']:.10g}\n"
        )
        write_array(stream, "expected_selected_points", point_selected[0])
        write_array(stream, "expected_mu", mu_selected)
        write_array(stream, "expected_lambda", lambda_selected)
        write_array(stream, "expected_points", point_selected)
        stream.write("END\n")


if __name__ == "__main__":
    main()
