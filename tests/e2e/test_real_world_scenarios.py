import pytest
import torch
import numpy as np
import os
import sys

# Add project root to sys.path to resolve imports cleanly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the scenario test logic from the verify script to ensure DRY (Don't Repeat Yourself) compliance
from tests.e2e.scripts.verify_scenarios import (
    test_intake_priming_and_compression,
    test_fixed_point_quantization_validation,
    test_clock_drift_wobble_compensation,
    test_sparse_multi_head_attention_routing,
    test_full_sequential_pipeline_integration
)

def test_scenario_1_intake_priming_and_atmospheric_compression():
    """Verify intake priming and compression in the MetatronCompressor pipeline."""
    test_intake_priming_and_compression()

def test_scenario_2_fixed_point_quantization_validation():
    """Verify fixed-point quantization model matches baseline within 0.20 deviation tolerance."""
    test_fixed_point_quantization_validation()

def test_scenario_3_clock_drift_wobble_compensation():
    """Verify clock-drift wobble compensation keeps attention and cooling softmax stable."""
    test_clock_drift_wobble_compensation()

def test_scenario_4_sparse_multi_head_attention_routing():
    """Verify sparse multi-head attention routing and void shunting to exhaust stats."""
    test_sparse_multi_head_attention_routing()

def test_scenario_5_full_sequential_pipeline_integration():
    """Verify full MetatronCompressor pipeline integration and BZS-GEMM backend routing."""
    test_full_sequential_pipeline_integration()

def test_real_world_scenarios_subprocess(subprocess_runner):
    """Run the verify_scenarios.py script in a subprocess to check standalone execution flow."""
    result = subprocess_runner("tests/e2e/scripts/verify_scenarios.py")
    assert result.returncode == 0
    assert "All Real-World Scenario test cases passed!" in result.stdout
