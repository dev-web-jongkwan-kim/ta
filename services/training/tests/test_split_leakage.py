import pandas as pd

from services.training.splits import SplitConfig, walk_forward_splits


def test_purged_walk_forward_no_overlap():
    idx = pd.date_range("2024-01-01", periods=10000, freq="1min", tz="UTC")
    cfg = SplitConfig(train_days=1, val_days=1, test_days=1, purge_bars=30, embargo_pct=0.1)
    for train_idx, val_idx, test_idx in walk_forward_splits(idx, cfg):
        assert train_idx.intersection(val_idx).empty
        assert train_idx.intersection(test_idx).empty
        assert val_idx.intersection(test_idx).empty
        if len(val_idx) and len(train_idx):
            assert train_idx.max() < val_idx.min()
        if len(test_idx) and len(val_idx):
            assert val_idx.max() < test_idx.min()
