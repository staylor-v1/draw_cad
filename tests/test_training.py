"""Tests for the src/training module."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools.step_analyzer import StepProperties
from src.training.ground_truth import StepGroundTruth, step_properties_to_ground_truth
from src.training.data_loader import TrainingDataIndex, TrainingPair, parse_pair_id
from src.training.manifest import load_manifest, save_manifest
from src.training.tiering import DifficultyTier, assign_tiers, classify_tier, compute_tier_distribution
from src.training.sampler import BenchmarkSampler, SamplingStrategy
from src.training.curriculum import CurriculumPhase, CurriculumScheduler
from src.training.fewshot_miner import FewShotMiner
from src.schemas.evaluation_result import EvaluationMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gt(
    face_count: int = 6,
    edge_count: int = 12,
    solid_count: int = 1,
    volume: float = 1000.0,
    bbox_extents: list[float] | None = None,
    is_valid: bool = True,
) -> StepGroundTruth:
    bbox_extents = bbox_extents or [10.0, 20.0, 30.0]
    return StepGroundTruth(
        volume=volume,
        surface_area=600.0,
        bounding_box=[[0, 0, 0], [10, 20, 30]],
        center_of_mass=[5, 10, 15],
        face_count=face_count,
        edge_count=edge_count,
        vertex_count=8,
        solid_count=solid_count,
        bbox_extents=bbox_extents,
        is_valid=is_valid,
    )


def _make_pair(
    pair_id: str = "12345_abcdef01_0001",
    tier: int | None = None,
    gt: StepGroundTruth | None = None,
) -> TrainingPair:
    return TrainingPair(
        pair_id=pair_id,
        part_id="12345",
        part_hash="abcdef01",
        view_number="0001",
        svg_path=Path(f"/fake/drawings_svg/{pair_id}.svg"),
        step_path=Path(f"/fake/shapes_step/{pair_id}.step"),
        png_path=Path(f"/fake/rendered_png/{pair_id}.png"),
        ground_truth=gt,
        tier=tier,
    )


def _make_index(n: int = 30, with_gt: bool = True, with_tiers: bool = True) -> TrainingDataIndex:
    pairs = []
    for i in range(n):
        gt = _make_gt(face_count=6 + i * 2, edge_count=12 + i * 3) if with_gt else None
        pair = _make_pair(
            pair_id=f"{10000 + i}_aabb{i:04x}_0000",
            gt=gt,
        )
        pairs.append(pair)

    idx = TrainingDataIndex(pairs=pairs)
    if with_tiers and with_gt:
        assign_tiers(idx)
    return idx


# ===========================================================================
# TestTrainingPair
# ===========================================================================

class TestTrainingPair:
    def test_filename_parsing(self):
        result = parse_pair_id("37605_e35cc4df_0007")
        assert result == ("37605", "e35cc4df", "0007")

    def test_filename_parsing_invalid(self):
        assert parse_pair_id("not-a-valid-name") is None

    def test_pair_fields(self):
        pair = _make_pair()
        assert pair.pair_id == "12345_abcdef01_0001"
        assert pair.part_id == "12345"
        assert pair.part_hash == "abcdef01"
        assert pair.view_number == "0001"


# ===========================================================================
# TestStepGroundTruth
# ===========================================================================

class TestStepGroundTruth:
    def test_conversion_from_step_properties(self):
        props = StepProperties(
            is_valid=True,
            volume=500.0,
            surface_area=300.0,
            bounding_box=[[0, 0, 0], [10, 20, 30]],
            center_of_mass=[5, 10, 15],
            face_count=12,
            edge_count=24,
            vertex_count=16,
            solid_count=1,
        )
        gt = step_properties_to_ground_truth(props)
        assert gt.volume == 500.0
        assert gt.face_count == 12
        assert gt.is_valid is True
        assert gt.bbox_extents == [10.0, 20.0, 30.0]

    def test_bbox_extents_sorted(self):
        props = StepProperties(
            is_valid=True,
            bounding_box=[[0, 0, 0], [30, 10, 20]],
        )
        gt = step_properties_to_ground_truth(props)
        assert gt.bbox_extents == [10.0, 20.0, 30.0]

    def test_serialisation_round_trip(self):
        gt = _make_gt()
        data = gt.model_dump()
        restored = StepGroundTruth.model_validate(data)
        assert restored == gt


# ===========================================================================
# TestTrainingDataIndex
# ===========================================================================

class TestTrainingDataIndex:
    def test_index_size(self):
        idx = _make_index(10)
        assert idx.size == 10

    def test_get_by_id(self):
        idx = _make_index(5)
        pair = idx.get_by_id(idx.pairs[0].pair_id)
        assert pair is not None
        assert pair.pair_id == idx.pairs[0].pair_id

    def test_get_by_id_missing(self):
        idx = _make_index(5)
        assert idx.get_by_id("nonexistent") is None

    def test_get_by_tier(self):
        idx = _make_index(30)
        tier_1 = idx.get_by_tier(1)
        assert all(p.tier == 1 for p in tier_1)

    def test_get_by_tiers(self):
        idx = _make_index(30)
        result = idx.get_by_tiers([1, 2])
        assert all(p.tier in (1, 2) for p in result)

    def test_directory_scanning(self):
        """Test from_directory with real training data if available."""
        root = Path("training_data")
        if not (root / "drawings_svg").is_dir():
            pytest.skip("training_data not available")
        idx = TrainingDataIndex.from_directory(root)
        assert idx.size > 0

    def test_rebuild_lookup(self):
        idx = _make_index(5)
        new_pair = _make_pair(pair_id="99999_ffff0000_0000")
        idx.pairs.append(new_pair)
        idx.rebuild_lookup()
        assert idx.get_by_id("99999_ffff0000_0000") is not None


# ===========================================================================
# TestSvgRenderer
# ===========================================================================

class TestSvgRenderer:
    @patch("src.training.svg_renderer.cairosvg")
    def test_render_svg_to_png(self, mock_cairo):
        from src.training.svg_renderer import render_svg_to_png

        with tempfile.TemporaryDirectory() as tmp:
            svg = Path(tmp) / "test.svg"
            svg.write_text("<svg></svg>")
            out = Path(tmp) / "out.png"
            result = render_svg_to_png(svg, out)
            assert result == out
            mock_cairo.svg2png.assert_called_once()

    def test_output_path_generation(self):
        """Output PNG should match input SVG basename."""
        from src.training.svg_renderer import render_svg_to_png

        with tempfile.TemporaryDirectory() as tmp:
            svg = Path(tmp) / "12345_abc_0001.svg"
            svg.write_text("<svg></svg>")
            out = Path(tmp) / "rendered" / "12345_abc_0001.png"
            with patch("src.training.svg_renderer.cairosvg"):
                result = render_svg_to_png(svg, out)
            assert result.stem == "12345_abc_0001"


# ===========================================================================
# TestTiering
# ===========================================================================

class TestTiering:
    def test_simple_tier(self):
        gt = _make_gt(face_count=6, edge_count=12, solid_count=1)
        assert classify_tier(gt) == DifficultyTier.SIMPLE

    def test_medium_tier(self):
        gt = _make_gt(face_count=20, edge_count=40, solid_count=1)
        assert classify_tier(gt) == DifficultyTier.MEDIUM

    def test_complex_tier(self):
        gt = _make_gt(face_count=50, edge_count=100, solid_count=3)
        assert classify_tier(gt) == DifficultyTier.COMPLEX

    def test_low_fill_ratio_promotes(self):
        """Very low volume-to-bbox ratio should bump up tier."""
        gt = _make_gt(
            face_count=8,
            edge_count=16,
            solid_count=1,
            volume=10.0,
            bbox_extents=[10.0, 20.0, 30.0],
        )
        # fill = 10 / 6000 = 0.0017 < 0.15 → should promote from SIMPLE to MEDIUM
        tier = classify_tier(gt)
        assert tier >= DifficultyTier.MEDIUM

    def test_high_edge_face_ratio_promotes(self):
        """High edge/face ratio should bump up tier."""
        gt = _make_gt(face_count=10, edge_count=50, solid_count=1)
        # edge/face = 5.0 > 4.0 → promote
        tier = classify_tier(gt)
        assert tier >= DifficultyTier.MEDIUM

    def test_tier_distribution(self):
        idx = _make_index(30)
        dist = compute_tier_distribution(idx)
        assert sum(dist.values()) == 30


# ===========================================================================
# TestBenchmarkSampler
# ===========================================================================

class TestBenchmarkSampler:
    def test_random_sample_size(self):
        idx = _make_index(50)
        sampler = BenchmarkSampler(idx, sample_size=20, strategy="random")
        result = sampler.sample()
        assert len(result) == 20

    def test_stratified_has_all_tiers(self):
        idx = _make_index(50)
        sampler = BenchmarkSampler(idx, sample_size=20, strategy="stratified")
        result = sampler.sample()
        tiers_present = {p.tier for p in result}
        # At least two tiers should be represented
        assert len(tiers_present) >= 2

    def test_sentinel_inclusion(self):
        idx = _make_index(50)
        sentinel_id = idx.pairs[0].pair_id
        sampler = BenchmarkSampler(
            idx,
            sample_size=20,
            strategy="random",
            sentinel_ids=[sentinel_id],
            sentinel_count=1,
        )
        result = sampler.sample()
        result_ids = {p.pair_id for p in result}
        assert sentinel_id in result_ids

    def test_failure_targeted(self):
        idx = _make_index(50)
        sampler = BenchmarkSampler(idx, sample_size=20, strategy="failure_targeted")
        # Record low scores for a few pairs
        for p in idx.pairs[:10]:
            sampler.record_scores({p.pair_id: 0.2})
        result = sampler.sample()
        assert len(result) == 20

    def test_tier_filter(self):
        idx = _make_index(50)
        sampler = BenchmarkSampler(idx, sample_size=10, strategy="stratified")
        result = sampler.sample(tier_filter=[1])
        assert all(p.tier == 1 for p in result if p.tier is not None)

    def test_empty_pool(self):
        idx = TrainingDataIndex(pairs=[])
        sampler = BenchmarkSampler(idx, sample_size=10)
        assert sampler.sample() == []


# ===========================================================================
# TestCurriculumScheduler
# ===========================================================================

class TestCurriculumScheduler:
    def test_initial_phase(self):
        idx = _make_index(50)
        sched = CurriculumScheduler(idx)
        assert sched.phase_name == "foundation"

    def test_phase_advancement(self):
        idx = _make_index(50)
        sched = CurriculumScheduler(idx)

        # Score high enough to advance
        assert sched.should_advance(0.7) is True
        advanced = sched.advance()
        assert advanced is True
        assert sched.phase_name == "intermediate"

    def test_max_iterations_advance(self):
        idx = _make_index(50)
        phases = [
            CurriculumPhase(name="p1", tiers=[1], max_iterations=2, min_score_to_advance=0.9),
            CurriculumPhase(name="p2", tiers=[1, 2]),
        ]
        sched = CurriculumScheduler(idx, phases=phases)

        # Get samples to bump the iteration counter
        sched.get_current_sample(0)
        sched.get_current_sample(1)

        assert sched.should_advance(0.3) is True  # below score, but iterations maxed

    def test_no_advance_from_final_phase(self):
        idx = _make_index(50)
        sched = CurriculumScheduler(idx)
        # Advance to last phase
        sched.advance()
        sched.advance()
        assert sched.phase_name == "advanced"
        assert sched.advance() is False

    def test_reset_on_advance(self):
        idx = _make_index(50)
        sched = CurriculumScheduler(idx)
        sched.get_current_sample(0)
        assert sched.phase_iteration == 1
        sched.advance()
        assert sched.phase_iteration == 0

    def test_get_current_sample(self):
        idx = _make_index(50)
        sched = CurriculumScheduler(idx)
        sample = sched.get_current_sample(0)
        assert len(sample) > 0
        # Foundation phase should only have tier 1
        for p in sample:
            if p.tier is not None:
                assert p.tier == 1


# ===========================================================================
# TestFewShotMiner
# ===========================================================================

class TestFewShotMiner:
    def test_score_threshold(self):
        idx = _make_index(5)
        with tempfile.TemporaryDirectory() as tmp:
            miner = FewShotMiner(idx, storage_dir=tmp)
            pair = idx.pairs[0]
            pair.ground_truth = _make_gt()

            low_metrics = EvaluationMetrics(composite_score=0.5, geometry_valid=1.0)
            assert miner.record_successful_run(pair, "code", low_metrics) is False

    def test_successful_record(self):
        idx = _make_index(5)
        with tempfile.TemporaryDirectory() as tmp:
            miner = FewShotMiner(idx, storage_dir=tmp)
            pair = idx.pairs[0]
            pair.ground_truth = _make_gt()

            metrics = EvaluationMetrics(composite_score=0.8, geometry_valid=1.0)
            result = miner.record_successful_run(pair, "from build123d import *", metrics)
            assert result is True
            assert miner.mined_count == 1

            # Check file was written
            md_path = Path(tmp) / f"{pair.pair_id}.md"
            assert md_path.exists()
            content = md_path.read_text()
            assert "build123d" in content

    def test_deduplication(self):
        idx = _make_index(5)
        with tempfile.TemporaryDirectory() as tmp:
            miner = FewShotMiner(idx, storage_dir=tmp)
            pair = idx.pairs[0]
            pair.ground_truth = _make_gt()

            metrics = EvaluationMetrics(composite_score=0.8, geometry_valid=1.0)
            miner.record_successful_run(pair, "code", metrics)
            second = miner.record_successful_run(pair, "code2", metrics)
            assert second is False
            assert miner.mined_count == 1

    def test_geometry_valid_required(self):
        idx = _make_index(5)
        with tempfile.TemporaryDirectory() as tmp:
            miner = FewShotMiner(idx, storage_dir=tmp)
            pair = idx.pairs[0]
            pair.ground_truth = _make_gt()

            metrics = EvaluationMetrics(composite_score=0.9, geometry_valid=0.5)
            assert miner.record_successful_run(pair, "code", metrics) is False


# ===========================================================================
# TestManifest
# ===========================================================================

class TestManifest:
    def test_save_and_load(self):
        idx = _make_index(5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            save_manifest(idx, path)

            loaded = load_manifest(path)
            assert loaded.size == 5
            assert loaded.pairs[0].pair_id == idx.pairs[0].pair_id

    def test_ground_truth_preserved(self):
        idx = _make_index(5, with_gt=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            save_manifest(idx, path)

            loaded = load_manifest(path)
            assert loaded.pairs[0].ground_truth is not None
            assert loaded.pairs[0].ground_truth.face_count == idx.pairs[0].ground_truth.face_count

    def test_tier_preserved(self):
        idx = _make_index(5, with_gt=True, with_tiers=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            save_manifest(idx, path)

            loaded = load_manifest(path)
            assert loaded.pairs[0].tier == idx.pairs[0].tier


# ===========================================================================
# Integration test (uses real training data if available)
# ===========================================================================

class TestIntegration:
    def test_load_real_pairs(self):
        """Load a subset of real training data and verify ground truth extraction."""
        root = Path("training_data")
        if not (root / "drawings_svg").is_dir():
            pytest.skip("training_data not available")

        idx = TrainingDataIndex.from_directory(root)
        assert idx.size > 0

        # Verify first 10 pairs have matching SVG/STEP
        for pair in idx.pairs[:10]:
            assert pair.svg_path.exists(), f"SVG missing: {pair.svg_path}"
            assert pair.step_path.exists(), f"STEP missing: {pair.step_path}"
            assert pair.part_id.isdigit()
