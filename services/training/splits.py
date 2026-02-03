from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import pandas as pd


@dataclass
class SplitConfig:
    train_days: int = 180
    val_days: int = 30
    test_days: int = 30
    purge_bars: int = 0
    embargo_pct: float = 0.0


def walk_forward_splits(index: pd.DatetimeIndex, cfg: SplitConfig) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]]:
    start = index.min()
    end = index.max()
    cursor = start
    while True:
        train_end = cursor + pd.Timedelta(days=cfg.train_days)
        val_end = train_end + pd.Timedelta(days=cfg.val_days)
        test_end = val_end + pd.Timedelta(days=cfg.test_days)
        if test_end > end:
            break
        train_idx = index[(index >= cursor) & (index < train_end)]
        val_idx = index[(index >= train_end) & (index < val_end)]
        test_idx = index[(index >= val_end) & (index < test_end)]
        yield _apply_purge_embargo(train_idx, val_idx, test_idx, cfg)
        cursor = val_end


def _apply_purge_embargo(
    train_idx: pd.DatetimeIndex,
    val_idx: pd.DatetimeIndex,
    test_idx: pd.DatetimeIndex,
    cfg: SplitConfig,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    if cfg.purge_bars > 0:
        purge_delta = pd.Timedelta(minutes=cfg.purge_bars)
        train_idx = train_idx[train_idx < (val_idx.min() - purge_delta)]
        val_idx = val_idx[val_idx < (test_idx.min() - purge_delta)]
    if cfg.embargo_pct > 0:
        embargo_delta = pd.Timedelta(seconds=int((test_idx.max() - test_idx.min()).total_seconds() * cfg.embargo_pct))
        val_idx = val_idx[val_idx < (test_idx.min() - embargo_delta)]
    return train_idx, val_idx, test_idx
