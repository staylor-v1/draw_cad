"""Tests for Gemma 4 Total_view_data experiment helpers."""
from __future__ import annotations

from src.experiments.gemma4_total_view import Gemma4TotalViewExperiment


def test_evenly_spaced_picks_extremes_and_middle():
    records = [{"case_id": f"{index:03d}"} for index in range(7)]

    selected = Gemma4TotalViewExperiment._evenly_spaced(records, 3)

    assert [item["case_id"] for item in selected] == ["000", "003", "006"]


def test_aggregate_case_results_computes_success_rate():
    experiment = object.__new__(Gemma4TotalViewExperiment)
    results = [
        {
            "success": True,
            "score": 0.4,
            "mean_visible_f1": 0.5,
            "mean_hidden_f1": 0.2,
            "duration_seconds": 2.0,
            "tool_calls": 1,
        },
        {
            "success": False,
            "score": 0.0,
            "mean_visible_f1": 0.0,
            "mean_hidden_f1": 0.0,
            "duration_seconds": 4.0,
            "tool_calls": 3,
        },
    ]

    aggregate = Gemma4TotalViewExperiment._aggregate_case_results(experiment, results)

    assert aggregate["total_cases"] == 2
    assert aggregate["successful_cases"] == 1
    assert aggregate["success_rate"] == 0.5
    assert aggregate["mean_score"] == 0.2
    assert aggregate["mean_tool_calls"] == 2.0
