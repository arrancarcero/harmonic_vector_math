import pytest
import os
import json

def test_infra_sanity_without_override(subprocess_runner):
    # Runs verify_infra_sanity.py with standard config
    result = subprocess_runner("tests/e2e/scripts/verify_infra_sanity.py")
    assert result.returncode == 0
    assert "base_frequency: 432.0" in result.stdout
    assert "strict_rules_enabled: True" in result.stdout
    assert "custom_e2e_test_key: None" in result.stdout

def test_infra_sanity_with_override(override_config, subprocess_runner):
    # Prepare custom configuration override
    custom_config = {
        "project": "Harmonic Engine / E2E Test Override",
        "hardware_heritage": {
            "base_frequency_hz": 999.0
        },
        "optimization_methods": {
            "1_strict_rules_meta_prompting": {
                "enabled": False
            }
        },
        "custom_e2e_test_key": "verified_value"
    }
    
    # Apply override
    override_config(custom_config)
    
    # Run the script
    result = subprocess_runner("tests/e2e/scripts/verify_infra_sanity.py")
    
    assert result.returncode == 0
    assert "base_frequency: 999.0" in result.stdout
    assert "strict_rules_enabled: False" in result.stdout
    assert "custom_e2e_test_key: verified_value" in result.stdout

def test_infra_sanity_restoration(subprocess_runner):
    # Verify that the config was restored to standard values
    result = subprocess_runner("tests/e2e/scripts/verify_infra_sanity.py")
    assert result.returncode == 0
    assert "base_frequency: 432.0" in result.stdout
    assert "strict_rules_enabled: True" in result.stdout
    assert "custom_e2e_test_key: None" in result.stdout
