import json
import os
import pytest
from click.testing import CliRunner
from resorter_py.cli import main

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def input_file(tmp_path):
    f = tmp_path / "items.csv"
    f.write_text("Apple\nBanana\nCherry\n")
    return str(f)

def test_cli_interactive_run(runner, input_file):
    """Test a short interactive ranking run."""
    # Simulate user input: 1 (Apple > Banana), 3 (Banana < Cherry), q (quit)
    result = runner.invoke(main, ["--input", input_file], input="1\n3\nq\n")
    
    assert result.exit_code == 0
    assert "Comparison 1/" in result.output
    assert "Is 'Apple' better than 'Banana'?" in result.output or "Is 'Banana' better than 'Apple'?" in result.output
    assert "Quitting..." in result.output
    assert "Item,Rank,Confidence,Uncertainty" in result.output
    assert "Apple" in result.output
    assert "Banana" in result.output
    assert "Cherry" in result.output

def test_cli_save_load_state(runner, input_file, tmp_path):
    """Test saving state to a file, then loading that state in a subsequent run."""
    state_file = tmp_path / "state.json"
    
    # First run: provide some input and save state
    # Input: 1 (Better), q (Quit)
    result1 = runner.invoke(main, ["--input", input_file, "--save-state", str(state_file)], input="1\nq\n")
    assert result1.exit_code == 0
    assert f"Saved state to {state_file}" in result1.output
    assert os.path.exists(state_file)
    
    # Second run: load state
    # Input: q (Quit immediately)
    result2 = runner.invoke(main, ["--input", input_file, "--load-state", str(state_file)], input="q\n")
    assert result2.exit_code == 0
    assert f"Loaded state from {state_file}" in result2.output

def test_cli_json_output(runner, input_file):
    """Test non-default output format (JSON)."""
    result = runner.invoke(main, ["--input", input_file, "--format", "json"], input="q\n")
    
    assert result.exit_code == 0
    # The output should contain the JSON string. 
    # Click might have some prefix/suffix if there are prints before the final output.
    # We need to find the JSON part.
    try:
        # Search for the start of the JSON object
        json_start = result.output.find('{')
        json_str = result.output[json_start:]
        data = json.loads(json_str)
        assert "rankings" in data
        assert "confidences" in data
        assert "metadata" in data
    except (ValueError, json.JSONDecodeError) as e:
        pytest.fail(f"Output was not valid JSON: {result.output}. Error: {e}")

def test_cli_invalid_state_file(runner, input_file, tmp_path):
    """Test edge case: invalid load-state JSON file."""
    invalid_state = tmp_path / "invalid.json"
    invalid_state.write_text("not a json")
    
    result = runner.invoke(main, ["--input", input_file, "--load-state", str(invalid_state)], input="q\n")
    
    assert result.exit_code == 0
    assert f"Invalid state file {invalid_state}, starting fresh." in result.output

def test_cli_quit_early(runner, input_file):
    """Test edge case: user quitting early."""
    result = runner.invoke(main, ["--input", input_file], input="q\n")
    
    assert result.exit_code == 0
    assert "Quitting..." in result.output
    assert "Item,Rank,Confidence,Uncertainty" in result.output
