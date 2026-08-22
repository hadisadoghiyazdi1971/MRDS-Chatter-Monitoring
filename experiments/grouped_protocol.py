#!/usr/bin/env python3
"""Utilities for condition-grouped chatter evaluation.

The independent deployment unit is a machining condition.  The four files
ending in R1--R4 are repeated recordings of that condition and must stay on
the same side of every outer train/test split.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


RECORDING_RE = re.compile(
    r"^(?P<state>[SU])_WP(?P<workpiece>[AB])_L(?P<length>[0-9.]+)"
    r"_DOC(?P<doc>[0-9.]+)_WOC(?P<woc>[0-9.]+)"
    r"_N(?P<speed>[0-9]+)_F(?P<feed>[0-9]+)_R(?P<repetition>[1-4])$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class GroupedFold:
    split: int
    repeat: int
    fold: int
    seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def parse_recording_id(recording_id: str) -> dict[str, object]:
    """Parse one archive file stem and derive its condition identifier."""
    match = RECORDING_RE.fullmatch(recording_id)
    if match is None:
        raise ValueError(f"Recording id does not match the declared archive schema: {recording_id}")
    fields = match.groupdict()
    condition_id = re.sub(r"_R[1-4]$", "", recording_id, flags=re.IGNORECASE)
    return {
        "recording_id": recording_id,
        "condition_id": condition_id,
        "repetition": int(fields["repetition"]),
        "state_from_name": fields["state"].upper(),
        "workpiece": fields["workpiece"].upper(),
        "length_mm": float(fields["length"]),
        "doc_mm": float(fields["doc"]),
        "woc_mm": float(fields["woc"]),
        "spindle_rpm": int(fields["speed"]),
        "feed_mm_min": int(fields["feed"]),
    }


def build_group_manifest(
    recording_ids: Sequence[str],
    labels: Sequence[int],
    expected_repetitions: int = 4,
) -> pd.DataFrame:
    """Build and validate the condition/repetition manifest."""
    if len(recording_ids) != len(labels):
        raise ValueError("recording_ids and labels must have equal length")
    rows: list[dict[str, object]] = []
    for global_index, (recording_id, label) in enumerate(zip(recording_ids, labels)):
        row = parse_recording_id(str(recording_id))
        row.update({"global_index": global_index, "label": int(label)})
        expected_label = 1 if row["state_from_name"] == "U" else 0
        if int(label) != expected_label:
            raise ValueError(f"Label/name mismatch for {recording_id}")
        rows.append(row)

    manifest = pd.DataFrame(rows).sort_values("global_index").reset_index(drop=True)
    failures: list[str] = []
    for condition_id, group in manifest.groupby("condition_id", sort=True):
        repetitions = sorted(group["repetition"].astype(int).tolist())
        expected = list(range(1, expected_repetitions + 1))
        if repetitions != expected:
            failures.append(f"{condition_id}: repetitions={repetitions}, expected={expected}")
        if group["label"].nunique() != 1:
            failures.append(f"{condition_id}: inconsistent labels")
    if failures:
        raise ValueError("Invalid condition groups:\n" + "\n".join(failures))
    return manifest


def assert_group_disjoint(
    train_idx: Sequence[int], test_idx: Sequence[int], groups: Sequence[str]
) -> None:
    train_set = set(np.asarray(groups, dtype=object)[np.asarray(train_idx, dtype=int)])
    test_set = set(np.asarray(groups, dtype=object)[np.asarray(test_idx, dtype=int)])
    overlap = sorted(train_set & test_set)
    if overlap:
        raise RuntimeError(f"Condition leakage detected: {overlap[:5]}")


def repeated_stratified_group_folds(
    labels: Sequence[int],
    groups: Sequence[str],
    seeds: Iterable[int],
    n_splits: int = 4,
) -> Iterator[GroupedFold]:
    """Yield reproducible repeated stratified group folds."""
    y = np.asarray(labels, dtype=np.int64)
    g = np.asarray(groups, dtype=object)
    x_dummy = np.zeros((len(y), 1), dtype=np.float64)
    split_id = 0
    for repeat, seed in enumerate(seeds):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=int(seed)
        )
        seen_test: list[int] = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x_dummy, y, g)):
            assert_group_disjoint(train_idx, test_idx, g)
            seen_test.extend(int(i) for i in test_idx)
            yield GroupedFold(
                split=split_id,
                repeat=repeat,
                fold=fold,
                seed=int(seed),
                train_idx=np.asarray(train_idx, dtype=np.int64),
                test_idx=np.asarray(test_idx, dtype=np.int64),
            )
            split_id += 1
        if sorted(seen_test) != list(range(len(y))):
            raise RuntimeError(f"Repeat {repeat} is not a complete group cross-validation partition")


def fold_manifest_rows(fold: GroupedFold, manifest: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for partition, indices in (("train", fold.train_idx), ("test", fold.test_idx)):
        for idx in indices:
            source = manifest.iloc[int(idx)]
            rows.append({
                "split": fold.split,
                "repeat": fold.repeat,
                "fold": fold.fold,
                "seed": fold.seed,
                "partition": partition,
                "global_index": int(idx),
                "recording_id": source["recording_id"],
                "condition_id": source["condition_id"],
                "repetition": int(source["repetition"]),
                "label": int(source["label"]),
            })
    return rows


def selection_group_diagnostics(
    selected_local: Sequence[int], train_groups: Sequence[str]
) -> dict[str, int]:
    selected = np.asarray(selected_local, dtype=np.int64)
    groups = np.asarray(train_groups, dtype=object)[selected]
    _, counts = np.unique(groups, return_counts=True)
    return {
        "n_unique_selected_conditions": int(len(counts)),
        "n_duplicate_condition_slots": int(np.sum(np.maximum(counts - 1, 0))),
        "max_selected_per_condition": int(counts.max(initial=0)),
    }
