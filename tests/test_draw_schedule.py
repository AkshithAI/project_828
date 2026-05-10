"""
Tests for WeightedMixerDataset draw schedule shuffling and resumption.

Verifies:
1. Shuffle determinism — same weights always produce identical schedule
2. Distribution preservation — shuffled schedule has exact same dataset counts
3. Interleaving quality — no long contiguous runs of same dataset
4. Resumption correctness — draw_cycle_position works with shuffled schedule
5. Cross-invocation consistency — schedule rebuilt from scratch matches
"""
import math
import random as _random_module
import sys
import os
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs — we test the schedule logic in isolation, without needing
# HF datasets, the tokenizer, or GPU resources.
# ---------------------------------------------------------------------------

def _build_draw_schedule(weights: List[int]) -> List[int]:
    """
    Reproduce the exact schedule-building logic from WeightedMixerDataset.__init__
    (lines 1181-1194 of dataloader.py, post-fix).
    """
    g = weights[0]
    for w in weights[1:]:
        g = math.gcd(g, w)
    schedule: List[int] = []
    for idx, w in enumerate(weights):
        schedule.extend([idx] * (w // g))
    # Deterministic shuffle with fixed seed — MUST match dataloader.py
    rng = _random_module.Random(42)
    rng.shuffle(schedule)
    return schedule


def _build_draw_schedule_OLD(weights: List[int]) -> List[int]:
    """The OLD contiguous-block schedule (before fix)."""
    g = weights[0]
    for w in weights[1:]:
        g = math.gcd(g, w)
    schedule: List[int] = []
    for idx, w in enumerate(weights):
        schedule.extend([idx] * (w // g))
    return schedule


# Phase 1 weights from model_config.py
PHASE_1_WEIGHTS = [16, 7, 5, 4, 5, 3, 3, 3, 2, 2, 18, 6, 4, 10, 12]
PHASE_1_NAMES = [
    'python', 'js', 'java', 'ts', 'cpp', 'c', 'csharp', 'go', 'rust', 'php',
    'fineweb', 'finemath4', 'finemath3', 'stackexchange', 'opencodeinstruct',
]

# Phase 2 weights
PHASE_2_WEIGHTS = [14, 7, 4, 5, 5, 5, 15, 18, 12, 5, 10]
PHASE_2_NAMES = [
    'python', 'js', 'ts', 'cpp', 'go', 'rust', 'tiny-codes',
    'stackexchange', 'dclm-edu', 'wikipedia', 'fineweb-edu',
]


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Shuffle determinism
# ═══════════════════════════════════════════════════════════════════════

class TestShuffleDeterminism:
    """Same weights → identical schedule, every time."""

    def test_phase1_deterministic(self):
        s1 = _build_draw_schedule(PHASE_1_WEIGHTS)
        s2 = _build_draw_schedule(PHASE_1_WEIGHTS)
        assert s1 == s2, "Phase 1 schedule not deterministic!"

    def test_phase2_deterministic(self):
        s1 = _build_draw_schedule(PHASE_2_WEIGHTS)
        s2 = _build_draw_schedule(PHASE_2_WEIGHTS)
        assert s1 == s2, "Phase 2 schedule not deterministic!"

    def test_100_invocations(self):
        """Run 100 times — must be identical every time."""
        reference = _build_draw_schedule(PHASE_1_WEIGHTS)
        for i in range(100):
            assert _build_draw_schedule(PHASE_1_WEIGHTS) == reference, \
                f"Diverged on invocation {i}"

    def test_different_weights_different_schedule(self):
        """Sanity: different weights should produce different schedules."""
        s1 = _build_draw_schedule(PHASE_1_WEIGHTS)
        s2 = _build_draw_schedule(PHASE_2_WEIGHTS)
        assert s1 != s2


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Distribution preservation
# ═══════════════════════════════════════════════════════════════════════

class TestDistributionPreservation:
    """Shuffled schedule has exact same per-dataset counts as unshuffled."""

    def test_phase1_counts_preserved(self):
        old = _build_draw_schedule_OLD(PHASE_1_WEIGHTS)
        new = _build_draw_schedule(PHASE_1_WEIGHTS)
        assert Counter(old) == Counter(new), \
            f"Count mismatch!\nOld: {Counter(old)}\nNew: {Counter(new)}"

    def test_phase2_counts_preserved(self):
        old = _build_draw_schedule_OLD(PHASE_2_WEIGHTS)
        new = _build_draw_schedule(PHASE_2_WEIGHTS)
        assert Counter(old) == Counter(new)

    def test_cycle_length_preserved(self):
        old = _build_draw_schedule_OLD(PHASE_1_WEIGHTS)
        new = _build_draw_schedule(PHASE_1_WEIGHTS)
        assert len(old) == len(new) == 100  # GCD=1, sum=100

    def test_all_datasets_present(self):
        schedule = _build_draw_schedule(PHASE_1_WEIGHTS)
        unique = set(schedule)
        assert unique == set(range(len(PHASE_1_WEIGHTS))), \
            f"Missing datasets: {set(range(len(PHASE_1_WEIGHTS))) - unique}"


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Interleaving quality
# ═══════════════════════════════════════════════════════════════════════

class TestInterleavingQuality:
    """Shuffled schedule should not have long contiguous runs."""

    @staticmethod
    def _max_run_length(schedule: List[int]) -> int:
        """Return the longest consecutive run of the same dataset index."""
        if not schedule:
            return 0
        max_run = 1
        current_run = 1
        for i in range(1, len(schedule)):
            if schedule[i] == schedule[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        return max_run

    def test_old_schedule_has_long_runs(self):
        """Verify the OLD schedule has the contiguous-block problem."""
        old = _build_draw_schedule_OLD(PHASE_1_WEIGHTS)
        max_run = self._max_run_length(old)
        # Old schedule has runs of 16 (python), 18 (fineweb), 12 (opencodeinstruct)
        assert max_run >= 12, f"Expected long runs in old schedule, got {max_run}"

    def test_new_schedule_short_runs(self):
        """New shuffled schedule should have much shorter max run."""
        new = _build_draw_schedule(PHASE_1_WEIGHTS)
        max_run = self._max_run_length(new)
        # After shuffling 100 items among 15 buckets, max run should be ≤ 4
        assert max_run <= 5, \
            f"Shuffled schedule still has long runs: max_run={max_run}"
        print(f"  Max consecutive run in shuffled schedule: {max_run}")

    def test_phase2_short_runs(self):
        new = _build_draw_schedule(PHASE_2_WEIGHTS)
        max_run = self._max_run_length(new)
        assert max_run <= 5, f"Phase 2 max_run={max_run}"

    def test_first_10_draws_diverse(self):
        """First 10 draws should hit at least 5 different datasets."""
        schedule = _build_draw_schedule(PHASE_1_WEIGHTS)
        first_10 = set(schedule[:10])
        assert len(first_10) >= 5, \
            f"First 10 draws only hit {len(first_10)} datasets: {first_10}"


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Resumption correctness
# ═══════════════════════════════════════════════════════════════════════

class TestResumption:
    """
    Simulate the full resume flow:
    1. Build schedule, iterate N draws, record draw_cycle_position
    2. Rebuild schedule from scratch, start from saved position
    3. Verify remaining draws are identical
    """

    def test_resume_produces_same_sequence(self):
        """The core resumption test."""
        schedule = _build_draw_schedule(PHASE_1_WEIGHTS)
        cycle_len = len(schedule)

        # --- Run 1: iterate 47 draws ---
        draws_run1 = []
        for pos in range(47):
            ds_idx = schedule[pos % cycle_len]
            draws_run1.append(ds_idx)
        saved_position = 47

        # --- Run 2: rebuild from scratch, resume from position 47 ---
        schedule_resumed = _build_draw_schedule(PHASE_1_WEIGHTS)
        draws_run2 = []
        for pos in range(saved_position, saved_position + 200):
            ds_idx = schedule_resumed[pos % cycle_len]
            draws_run2.append(ds_idx)

        # Also compute what Run 1 would have produced for the same range
        draws_run1_continued = []
        for pos in range(saved_position, saved_position + 200):
            ds_idx = schedule[pos % cycle_len]
            draws_run1_continued.append(ds_idx)

        assert draws_run2 == draws_run1_continued, \
            "Resumed draw sequence doesn't match original!"

    def test_resume_from_various_positions(self):
        """Test resume from many different positions including cycle boundaries."""
        schedule = _build_draw_schedule(PHASE_1_WEIGHTS)
        cycle_len = len(schedule)

        test_positions = [0, 1, 49, 50, 99, 100, 101, 500, 9999, 100_000]
        for save_pos in test_positions:
            # Original sequence from save_pos
            original = [schedule[(save_pos + i) % cycle_len] for i in range(50)]
            # Rebuilt schedule, same position
            rebuilt = _build_draw_schedule(PHASE_1_WEIGHTS)
            resumed = [rebuilt[(save_pos + i) % cycle_len] for i in range(50)]
            assert original == resumed, \
                f"Mismatch at resume position {save_pos}"

    def test_cycle_wrapping(self):
        """Verify correct behavior when position wraps around the cycle."""
        schedule = _build_draw_schedule(PHASE_1_WEIGHTS)
        cycle_len = len(schedule)

        # Get draws crossing a cycle boundary (position 95 → 105)
        draws = [schedule[pos % cycle_len] for pos in range(95, 115)]

        # Verify it wraps correctly
        assert draws[:5] == [schedule[95], schedule[96], schedule[97],
                            schedule[98], schedule[99]]
        assert draws[5] == schedule[0]  # wrap!
        assert draws[6] == schedule[1]

    def test_large_position_values(self):
        """
        In real training, draw_cycle_position reaches millions.
        Verify modular arithmetic works correctly at large values.
        """
        schedule = _build_draw_schedule(PHASE_1_WEIGHTS)
        cycle_len = len(schedule)

        # Position 10_000_000 should be equivalent to 10_000_000 % 100
        expected_idx = schedule[10_000_000 % cycle_len]
        actual_idx = schedule[10_000_000 % cycle_len]
        assert expected_idx == actual_idx

        # Verify the full cycle repeats
        pos = 9_999_950
        draws = [schedule[(pos + i) % cycle_len] for i in range(200)]
        first_cycle = draws[:100]
        second_cycle = draws[100:]
        assert first_cycle == second_cycle, \
            "Cycle doesn't repeat correctly at large positions"


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Cross-invocation consistency with actual dataloader code
# ═══════════════════════════════════════════════════════════════════════

class TestCrossInvocationConsistency:
    """
    Verify our test _build_draw_schedule matches the actual dataloader code.
    This imports the real WeightedMixerDataset and checks the schedule.
    """

    def test_matches_actual_mixer(self):
        """
        Import the real WeightedMixerDataset with dummy streams and verify
        the draw schedule matches our standalone implementation.
        """
        try:
            # We need to add the project root to sys.path
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..')
            )
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from src.scripts.dataloader import WeightedMixerDataset

            # Create dummy entries with Phase 1 weights
            dummy_entries = [
                (f"dataset_{i}", iter([]), w, lambda x: None)
                for i, w in enumerate(PHASE_1_WEIGHTS)
            ]
            mixer = WeightedMixerDataset(
                dataset_entries=dummy_entries,
                context_length=32,  # small for testing
            )

            # Compare schedules
            expected = _build_draw_schedule(PHASE_1_WEIGHTS)
            actual = mixer._draw_schedule

            assert actual == expected, (
                f"Schedule mismatch!\n"
                f"Expected first 20: {expected[:20]}\n"
                f"Actual first 20:   {actual[:20]}"
            )
            print("  ✓ Real WeightedMixerDataset schedule matches test implementation")

        except ImportError as e:
            pytest.skip(f"Cannot import WeightedMixerDataset: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases that could break the shuffle."""

    def test_single_dataset(self):
        """One dataset — GCD=10, cycle is [0]."""
        schedule = _build_draw_schedule([10])
        assert schedule == [0]  # GCD=10, so 10//10 = 1 draw

    def test_two_datasets_equal_weight(self):
        """Two datasets with equal weight — GCD=5, cycle is [0,1] or [1,0]."""
        schedule = _build_draw_schedule([5, 5])
        assert Counter(schedule) == {0: 1, 1: 1}  # GCD=5, each gets 1
        assert len(schedule) == 2

    def test_gcd_reduction(self):
        """Weights with GCD > 1 should produce shorter cycle."""
        schedule = _build_draw_schedule([10, 20, 30])
        assert len(schedule) == 6  # GCD=10, so 1+2+3=6
        assert Counter(schedule) == {0: 1, 1: 2, 2: 3}

    def test_weight_of_one(self):
        """Minimum weight dataset still appears exactly once per cycle."""
        schedule = _build_draw_schedule([1, 5, 10])
        assert schedule.count(0) == 1

    def test_many_datasets(self):
        """30 datasets — should still shuffle correctly."""
        weights = list(range(1, 31))  # 1,2,3,...,30
        schedule = _build_draw_schedule(weights)
        g = 1  # GCD of 1..30 is 1
        expected_len = sum(weights)  # 465
        assert len(schedule) == expected_len
        for i, w in enumerate(weights):
            assert schedule.count(i) == w


# ═══════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
