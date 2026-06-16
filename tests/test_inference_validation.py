from __future__ import annotations

import pandas as pd

from src.inference import validate_input_features


def test_missing_features_fails() -> None:
    feature_columns = ["a", "b"]
    input_df = pd.DataFrame([{ "a": 1.0 }])

    result = validate_input_features(input_df, feature_columns)

    assert result["validation_status"] is False
    assert result["missing_features"] == ["b"]


def test_extra_features_are_ignored_with_warning() -> None:
    feature_columns = ["a", "b"]
    input_df = pd.DataFrame([{ "a": 1.0, "b": 2.0, "extra": 99.0 }])

    result = validate_input_features(input_df, feature_columns)

    assert result["validation_status"] is True
    assert result["extra_features"] == ["extra"]
    assert list(result["ordered_df"].columns) == feature_columns


def test_non_numeric_features_fail() -> None:
    feature_columns = ["a", "b"]
    input_df = pd.DataFrame([{ "a": 1.0, "b": "bad" }])

    result = validate_input_features(input_df, feature_columns)

    assert result["validation_status"] is False
    assert result["non_numeric_features"] == ["b"]


def test_columns_are_ordered_exactly() -> None:
    feature_columns = ["b", "a"]
    input_df = pd.DataFrame([{ "a": 1.0, "b": 2.0 }])

    result = validate_input_features(input_df, feature_columns)

    assert result["validation_status"] is True
    assert list(result["ordered_df"].columns) == ["b", "a"]
