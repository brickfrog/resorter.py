import json
import os
import pytest
from click.testing import CliRunner
from resorter_py.cli import main

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def input_csv(tmp_path):
    csv_file = tmp_path / "items.csv"
    csv_file.write_text("Item\nApple\nBanana\nCherry")
    return str(csv_file)

def test_cli_interactive_run(runner, input_csv):
    """Test a short interactive ranking run."""
    # Simulate user input: 1 (Apple > Banana), 1 (Apple > Cherry), q (Quit)
    result = runner.invoke(main, ["--input", input_csv], input="1\n1\nq\n")
    
    assert result.exit_code == 0
    assert "Number of queries:" in result.output
    assert "Comparison 1/" in result.output
    assert "Quitting..." in result.output
    assert "Apple" in result.output
    assert "Banana" in result.output
    assert "Cherry" in result.output

def test_cli_save_load_state(runner, input_csv, tmp_path):
    """Test saving state and loading it in a subsequent run."""
    state_file = str(tmp_path / "state.json")
    
    # First run: save state
    # Input: 1 (Apple > Banana), q (Quit)
    runner.invoke(main, ["--input", input_csv, "--save-state", state_file], input="1\nq\n")
    
    assert os.path.exists(state_file)
    with open(state_file, "r") as f:
        state = json.load(f)
        assert "items" in state
        assert "Apple" in state["items"]

    # Second run: load state
    # Input: q (Quit immediately)
    result = runner.invoke(main, ["--input", input_csv, "--load-state", state_file], input="q\n")
    
    assert result.exit_code == 0
    assert f"Loaded state from {state_file}" in result.output
    # Verify it still has the rankings from the first run
    assert "Apple" in result.output
    assert "Banana" in result.output

def test_cli_json_output(runner, input_csv):
    """Test JSON output format."""
    # Input: q (Quit immediately)
    result = runner.invoke(main, ["--input", input_csv, "--format", "json"], input="q\n")
    
    assert result.exit_code == 0
    # The output might have some text before the JSON if there are print statements
    # but CliRunner.invoke captures everything. We need to find the JSON part.
    # Looking at cli.py, it prints "Number of queries", "Comparison commands", "Quitting..."
    # and then the JSON.
    
    output_lines = result.output.splitlines()
    json_str = ""
    start_json = False
    for line in output_lines:
        if line.strip() == "{":
            start_json = True
        if start_json:
            json_str += line + "\n"
    
    data = json.loads(json_str)
    assert "rankings" in data
    assert "confidences" in data
    assert "metadata" in data
    assert "Apple" in data["rankings"]

def test_cli_invalid_load_state(runner, input_csv, tmp_path):
    """Test edge case: loading an invalid JSON state file."""
    invalid_state = tmp_path / "invalid.json"
    invalid_state.write_text("not a json")
    
    # Input: q (Quit immediately)
    result = runner.invoke(main, ["--input", input_csv, "--load-state", str(invalid_state)], input="q\n")
    
    assert result.exit_code == 0
    assert f"Invalid state file {invalid_state}, starting fresh." in result.output

def test_cli_quit_early(runner, input_csv):
    """Test edge case: quitting early."""
    # Input: q (Quit immediately on first question)
    result = runner.invoke(main, ["--input", input_csv], input="q\n")
    
    assert result.exit_code == 0
    assert "Quitting..." in result.output
    # Should still show rankings even if quit early
    assert "Item,Rank,Confidence,Uncertainty" in result.output
    assert "Apple" in result.output

