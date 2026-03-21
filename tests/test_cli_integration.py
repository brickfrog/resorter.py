import json
import pytest
from click.testing import CliRunner
from resorter_py.cli import main

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def input_file(tmp_path):
    f = tmp_path / "items.csv"
    f.write_text("Item\nApple\nBanana\nCherry")
    return f

def test_interactive_ranking(runner, input_file):
    """Test a short interactive ranking session."""
    inputs = ["1", "1", "q"]
    result = runner.invoke(main, ["--input", str(input_file)], input="\n".join(inputs) + "\n")
    
    assert result.exit_code == 0, result.output
    assert "Number of queries:" in result.output
    assert "Apple" in result.output
    assert "Banana" in result.output
    assert "Cherry" in result.output

def test_save_load_state(runner, input_file, tmp_path):
    """Test saving state to a file and loading it in a subsequent run."""
    state_file = tmp_path / "state.json"
    
    # Run 1: Save state after 1 comparison
    inputs1 = ["1", "q"]
    result1 = runner.invoke(main, ["--input", str(input_file), "--save-state", str(state_file)], input="\n".join(inputs1) + "\n")
    assert result1.exit_code == 0, result1.output
    assert state_file.exists()
    
    # Run 2: Load state and continue
    inputs2 = ["1", "q"]
    result2 = runner.invoke(main, ["--input", str(input_file), "--load-state", str(state_file)], input="\n".join(inputs2) + "\n")
    assert result2.exit_code == 0, result2.output
    assert f"Loaded state from {state_file}" in result2.output

def test_json_output(runner, input_file):
    """Test requesting JSON output format and validate the result."""
    inputs = ["1", "1", "q"]
    result = runner.invoke(main, ["--input", str(input_file), "--format", "json"], input="\n".join(inputs) + "\n")
    
    assert result.exit_code == 0, result.output
    
    # Find the JSON part
    output_lines = result.output.splitlines()
    json_str = ""
    start_json = False
    for line in output_lines:
        if line.strip() == "{":
            start_json = True
        if start_json:
            json_str += line + "\n"
            
    assert json_str, f"Could not find JSON output in: {result.output}"
    data = json.loads(json_str)
    assert "rankings" in data
    assert "confidences" in data
    assert "metadata" in data
    assert "Apple" in data["rankings"]

def test_invalid_load_state(runner, input_file, tmp_path):
    """Test loading an invalid JSON state file."""
    invalid_state = tmp_path / "invalid.json"
    invalid_state.write_text("not json")
    
    inputs = ["q"]
    result = runner.invoke(main, ["--input", str(input_file), "--load-state", str(invalid_state)], input="\n".join(inputs) + "\n")
    
    assert result.exit_code == 0, result.output
    assert f"Invalid state file {invalid_state}, starting fresh." in result.output

def test_quit_early(runner, input_file):
    """Test quitting the session early with 'q'."""
    inputs = ["q"]
    result = runner.invoke(main, ["--input", str(input_file)], input="\n".join(inputs) + "\n")
    
    assert result.exit_code == 0, result.output
    assert "Quitting..." in result.output
    assert "Item,Rank,Confidence,Uncertainty" in result.output
