from __future__ import annotations

import pandas as pd
import pytest

from src.simulation import apply_modifications


def test_pct_modification_applies() -> None:
    base_row = pd.DataFrame([{ "Amina Flow": 100.0 }])
    scenario, warnings, modified = apply_modifications(
        base_row,
        {"Amina Flow": {"mode": "pct", "value": 5}},
    )

    assert scenario.iloc[0]["Amina Flow"] == 105.0
    assert warnings == []
    assert modified[0]["new_value"] == 105.0


def test_abs_delta_modification_applies() -> None:
    base_row = pd.DataFrame([{ "Ore Pulp pH": 9.0 }])
    scenario, warnings, modified = apply_modifications(
        base_row,
        {"Ore Pulp pH": {"mode": "abs_delta", "value": 0.2}},
    )

    assert scenario.iloc[0]["Ore Pulp pH"] == 9.2
    assert warnings == []
    assert modified[0]["new_value"] == 9.2


def test_lag_warning_is_returned() -> None:
    base_row = pd.DataFrame([{ "% Silica Concentrate_lag_1": 2.0 }])
    _, warnings, _ = apply_modifications(
        base_row,
        {"% Silica Concentrate_lag_1": {"mode": "abs_delta", "value": 0.2}},
    )

    assert "This is a lab trend scenario, not an operational lever." in warnings


def test_more_than_one_row_raises() -> None:
    base_row = pd.DataFrame([
        {"Amina Flow": 100.0},
        {"Amina Flow": 110.0},
    ])

    with pytest.raises(ValueError):
        apply_modifications(base_row, {"Amina Flow": {"mode": "pct", "value": 5}})
