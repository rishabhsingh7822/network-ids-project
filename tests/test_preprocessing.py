# tests/test_preprocessing.py
import pytest
import pandas as pd
import numpy as np
from src.preprocessing.pipeline import build_pipeline
from src.preprocessing.feature_engineer import engineer_features

def test_pipeline_handles_nulls():
    X = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
    pipe = build_pipeline()
    result = pipe.fit_transform(X)
    assert not np.isnan(result).any(), "Pipeline should remove all NaN values"

def test_pipeline_scales_features():
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [100, 200, 300]})
    pipe = build_pipeline()
    result = pipe.fit_transform(X)
    # After StandardScaler mean should be ~0
    assert abs(result.mean()) < 1e-10, "Mean should be ~0 after scaling"

def test_pipeline_output_shape():
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    pipe = build_pipeline()
    result = pipe.fit_transform(X)
    assert result.shape == (3, 3), "Output shape should match input shape"

def test_feature_engineer_adds_columns():
    X = pd.DataFrame({
        'Total Length of Fwd Packets': [100, 200, 300],
        'Bwd Packet Length Max': [50, 100, 150],
        'Packet Length Mean': [10, 20, 30],
        'Packet Length Max': [20, 40, 60],
    })
    result = engineer_features(X)
    assert 'bytes_ratio' in result.columns, "bytes_ratio feature should be added"
    assert 'pkt_size_entropy' in result.columns, "pkt_size_entropy should be added"

def test_pipeline_handles_zeros():
    X = pd.DataFrame({'a': [0, 0, 0], 'b': [0, 0, 0]})
    pipe = build_pipeline()
    result = pipe.fit_transform(X)
    assert not np.isnan(result).any(), "Should handle all-zero features"