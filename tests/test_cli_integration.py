import os
import json
import pytest
import csv
from click.testing import CliRunner
from resorter_py.cli import main

@pytest.fixture
def input_csv(tmp_path):
    csv_path = tmp_path / "input.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Item"])
        writer.writerow(["Apple"])
        writer.writerow(["Banana"])
        writer.writerow(["Cherry"])
    return str(csv_path)

def test_cli_interactive_run(input_csv):
    """Scenario 1: One short interactive ranking run (simulate user input)"""
    runner = CliRunner()
    # Simulate: Apple > Banana (1), Banana > Cherry (1), then quit (q)
    result = runner.invoke(main, ["--input", input_csv], input="1\n1\nq\n")
    
    assert result.exit_code == 0
    assert "Number of queries:" in result.output
    assert "Comparison 1/" in result.output
    assert "Quitting..." in result.output
    assert "Apple" in result.output
    assert "Banana" in result.output
    assert "Cherry" in result.output

def test_cli_save_load_state(input_csv, tmp_path):
    """Scenario 2: Save state to a file, then load that state in a subsequent run"""
    state_file = str(tmp_path / "state.json")
    runner = CliRunner()
    
    # Run 1: Make one comparison (Apple > Banana) and save state
    # Input: 1 (yes), q (quit)
    result1 = runner.invoke(main, ["--input", input_csv, "--save-state", state_file], input="1\nq\n")
    assert result1.exit_code == 0
    assert os.path.exists(state_file)
    
    # Run 2: Load state and verify it continues (iteration_count should be in state)
    # Input: q (quit)
    # We want to check if it loaded correctly. 
    # The diagnostics show total comparisons if --diagnostics is used.
    result2 = runner.invoke(main, ["--input", input_csv, "--load-state", state_file, "--diagnostics"], input="q\n")
    assert result2.exit_code == 0
    assert f"Loaded state from {state_file}" in result2.output
    # Check that diagnostics show 1 comparison from the loaded state
    assert "Total comparisons: 1" in result2.output

def test_cli_json_output(input_csv, tmp_path):
    """Scenario 3: One non-default output format (JSON)"""
    output_json = tmp_path / "output.json"
    runner = CliRunner()
    # Just quit immediately to get output
    result = runner.invoke(main, ["--input", input_csv, "--format", "json", "--output", str(output_json)], input="q\n")
    
    assert result.exit_code == 0
    assert output_json.exists()
    
    with open(output_json, "r") as f:
        data = json.load(f)
    
    assert "rankings" in data
    assert "confidences" in data
    assert "metadata" in data
    assert data["metadata"]["total_comparisons"] == 0 # because we quit immediately
    
def test_cli_edge_cases(input_csv, tmp_path):
    """Scenario 4: One edge case: invalid load-state JSON file, and one for quitting early"""
    runner = CliRunner()
    
    # Invalid load-state JSON
    invalid_state = tmp_path / "invalid.json"
    invalid_state.write_text("not a json")
    
    result_invalid = runner.invoke(main, ["--input", input_csv, "--load-state", str(invalid_state)], input="q\n")
    assert result_invalid.exit_code == 0
    assert f"Invalid state file {invalid_state}, starting fresh." in result_invalid.output
    
    # Quitting early ('q')
    result_quit = runner.invoke(main, ["--input", input_csv], input="q\n")
    assert result_quit.exit_code == 0
    assert "Quitting..." in result_quit.output
