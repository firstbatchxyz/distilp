"""
Integration test for profiler and solver workflow.

This test demonstrates the complete workflow:
1. Profile a device using the profiler API
2. Profile a model using the profiler API
3. Load the profiles using the solver's loader
4. Run the HALDA solver with the profiles
"""

import tempfile
from pathlib import Path

import pytest

from distilp.profiler.api import profile_device, profile_model
from distilp.solver.components.loader import load_devices_and_model
from distilp.solver import halda_solve


@pytest.fixture
def test_repo_id():
    """Use a small model for fast testing."""
    return "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture
def device_profile_json(test_repo_id):
    """
    Profile the current device and return the profile as JSON.

    This uses the profiler API to generate a device profile
    that matches what would be output by the profiler CLI.
    """
    # Profile device with a small batch exponent for speed
    device_profile = profile_device(
        repo_id=test_repo_id,
        max_batch_exp=2,  # Only profile up to batch size 4 (2^2)
        debug=0,
    )

    # Convert to JSON (mimics CLI output)
    return device_profile.model_dump_json(indent=2)


@pytest.fixture
def model_profile_json(test_repo_id):
    """
    Profile the test model and return the profile as JSON.

    This uses the profiler API to generate a model profile
    that matches what would be output by the profiler CLI.
    """
    # Profile model with limited batch sizes for speed
    model_profile = profile_model(
        repo_id=test_repo_id,
        batch_sizes=[1, 2],  # Small batch sizes for speed
        sequence_length=128,  # Short sequence for speed
        debug=0,
    )

    # Convert to JSON (mimics CLI output)
    return model_profile.model_dump_json(indent=2)


def test_profile_and_solve_workflow(device_profile_json, model_profile_json):
    """
    Test the complete workflow from profiling to solving.

    This test:
    1. Profiles a device (via fixture)
    2. Profiles a model (via fixture)
    3. Loads the profiles using the solver's loader
    4. Runs the HALDA solver
    5. Validates the solution
    """
    # Create temporary files for the profiles (mimics CLI workflow)
    with tempfile.TemporaryDirectory() as tmpdir:
        device_path = Path(tmpdir) / "device_profile.json"
        model_path = Path(tmpdir) / "model_profile.json"

        # Write profiles to files (as CLI would do)
        device_path.write_text(device_profile_json)
        model_path.write_text(model_profile_json)

        # Load profiles using solver's loader (as solver CLI does)
        devices, model = load_devices_and_model(
            device_files=[str(device_path)],
            model_file=str(model_path),
        )

        # Validate loaded profiles
        assert len(devices) == 1, "Should have loaded 1 device"
        assert devices[0].name != "", "Device should have a name"
        assert model.L > 0, "Model should have layers"
        assert model.V > 0, "Model should have vocabulary"

        # Run HALDA solver (as solver CLI does)
        result = halda_solve(
            devs=devices,
            model=model,
            mip_gap=1e-4,
            plot=False,  # Disable plotting in test
            kv_bits="4bit",
        )

        # Validate solution
        assert result is not None, "Solver should return a result"
        assert result.k > 0, "Solution should have k > 0"
        assert result.obj_value >= 0, "Objective value should be non-negative"
        assert len(result.w) == len(devices), "Should have layer assignment for each device"
        assert sum(result.w) == model.L, "All layers should be assigned"

        # Print solution for visibility
        print(f"\nSolution: k={result.k}, objective={result.obj_value:.6f}")
        print(f"Layer distribution: {result.w}")


def test_profile_device_returns_valid_profile(test_repo_id):
    """Test that device profiling returns a valid DeviceProfile."""
    device_profile = profile_device(
        repo_id=test_repo_id,
        max_batch_exp=2,
        debug=0,
    )

    # Check required fields are populated
    assert device_profile.name != "", "Device should have a name"
    assert device_profile.os_type != "", "Device should have OS type"
    assert device_profile.d_avail_ram > 0, "Device should have available RAM"
    assert len(device_profile.scpu) > 0, "Device should have CPU throughput tables"

    # Check throughput table structure
    for dtype in ["f32", "fp16", "bf16"]:
        assert dtype in device_profile.scpu, f"Should have {dtype} throughput"
        assert len(device_profile.scpu[dtype]) > 0, f"{dtype} should have batch entries"


def test_profile_model_returns_valid_profile(test_repo_id):
    """Test that model profiling returns a valid ModelProfileSplit."""
    model_profile = profile_model(
        repo_id=test_repo_id,
        batch_sizes=[1, 2],
        sequence_length=128,
        debug=0,
    )

    # Check required fields are populated
    assert model_profile.L > 0, "Model should have layers"
    assert model_profile.V > 0, "Model should have vocabulary"
    assert model_profile.e_embed > 0, "Model should have embedding dimension"
    assert len(model_profile.b) == model_profile.L + 1, "Should have bytes for all layers"
    assert "prefill" in model_profile.f_q, "Should have prefill phase"
    assert "decode" in model_profile.f_q, "Should have decode phase"

    # Check batch sizes are present
    for batch_size in [1, 2]:
        batch_key = f"b_{batch_size}"
        assert batch_key in model_profile.f_q["prefill"], f"Should have prefill for {batch_key}"
        assert batch_key in model_profile.f_q["decode"], f"Should have decode for {batch_key}"


def test_existing_profile_folder(tmpdir):
    """
    Test loading from an existing profile folder.

    This tests the load_from_profile_folder function which is used
    by the solver CLI with --profile flag.
    """
    from distilp.solver.components.loader import load_from_profile_folder

    # Use the existing hermes_70b profile folder for testing
    profile_folder = "test/profiles/hermes_70b"

    # Load profiles
    devices, model = load_from_profile_folder(profile_folder)

    # Validate
    assert len(devices) > 0, "Should load at least one device"
    assert model.L > 0, "Model should have layers"

    # Run solver with loaded profiles
    result = halda_solve(
        devs=devices,
        model=model,
        mip_gap=1e-4,
        plot=False,
        kv_bits="4bit",
    )

    assert result is not None, "Solver should return a result"
    assert result.k > 0, "Solution should have k > 0"
    print(f"\nExisting profile solution: k={result.k}, objective={result.obj_value:.6f}")


if __name__ == "__main__":
    # Run tests manually for debugging
    pytest.main([__file__, "-v", "-s"])
