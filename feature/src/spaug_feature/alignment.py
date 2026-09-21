"""Align feature arrays to observation identifiers."""
import numpy as np
import pandas as pd


def align_observations(values, source_names, target_names):
    """Return rows in target order, with explicit identifier validation."""
    values = np.asarray(values)
    source = pd.Index(source_names).astype(str)
    target = pd.Index(target_names).astype(str)
    if not source.is_unique or not target.is_unique:
        raise ValueError("Observation identifiers must be unique for feature alignment")
    if values.ndim == 0 or len(values) != len(source):
        raise ValueError("Each feature row requires one source observation identifier")
    indices = source.get_indexer(target)
    if (indices < 0).any():
        raise ValueError(f"Feature cache requires {int((indices < 0).sum())} additional observation identifiers")
    return values[indices]


def match_sample_barcodes(raw_names, sample_id, patch_names):
    """Match patch barcodes to exact AnnData names, preserving patch order."""
    names = pd.Index(raw_names).astype(str)
    prefix = f"{sample_id}:"
    lookup = {}
    for name in names:
        barcode = name[len(prefix):] if name.startswith(prefix) else name
        if barcode in lookup:
            raise ValueError("Each sample requires unique observation barcodes")
        lookup[barcode] = name
    selected, matched = [], []
    seen = set()
    for index, barcode in enumerate(map(str, patch_names)):
        if barcode in seen:
            raise ValueError("Patch barcodes must be unique within a sample")
        seen.add(barcode)
        if barcode in lookup:
            selected.append(index)
            matched.append(lookup[barcode])
    return np.asarray(selected, dtype=int), matched
