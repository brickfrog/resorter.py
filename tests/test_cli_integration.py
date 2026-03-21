import json
import pytest
from click.testing import CliRunner
from resorter_py.cli import main

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def input_csv(tmp_path):
    csv_file = tmp_path / "items.csv"
    csv_file.write_text("Apple\nBanana\nCherry\n")
    return str(csv_file)

def test_interactive_ranking_run(runner, input_csv):
    """A short interactive ranking run (feed simulated input to the CLI, verify it completes and produces output)."""
    # Simulate answering '1' (first is better) for a few questions, then it should finish or we can quit.
    # We'll set --queries to a small number to ensure it finishes quickly.
    result = runner.invoke(main, ["--input", input_csv, "--queries", "2"], input="1\n1\n")
    
    assert result.exit_code == 0
    assert "Number of queries: 2" in result.output
    assert "Comparison 1/2" in result.output
    assert "Comparison 2/2" in result.output
    assert "Item,Rank,Confidence,Uncertainty" in result.output

def test_save_load_state(runner, input_csv, tmp_path):
    """Save state to a file, then load that state in a subsequent run (verify state persistence works end-to-end)."""
    state_file = tmp_path / "state.json"
    
    # First run: save state after 1 comparison
    result1 = runner.invoke(
        main, 
        ["--input", input_csv, "--queries", "2", "--save-state", str(state_file)], 
        input="1\nq\n"
    )
    assert result1.exit_code == 0
    assert state_file.exists()
    
    # Second run: load state and verify it acknowledges it
    result2 = runner.invoke(
        main, 
        ["--input", input_csv, "--queries", "2", "--load-state", str(state_file)], 
        input="q\n"
    )
    assert result2.exit_code == 0
    assert f"Loaded state from {state_file}" in result2.output

def test_json_output_format(runner, input_csv):
    """A non-default output format like JSON (pass --format json, verify valid JSON output)."""
    # Use --queries 1 to get to output quickly
    result = runner.invoke(main, ["--input", input_csv, "--queries", "1", "--format", "json"], input="1\n")
    
    assert result.exit_code == 0
    # Try to parse the output as JSON. We need to find where the JSON starts because of the headers/prompts.
    # Usually the JSON is at the end or we can search for '{'.
    # In cli.py, it prints JSON at the end if no --output is provided.
    
    output = result.output
    json_start = output.find('{')
    assert json_start != -1
    json_str = output[json_start:]
    
    data = json.loads(json_str)
    assert "rankings" in data
    assert "confidences" in data
    assert "metadata" in data
    assert len(data["rankings"]) == 3

def test_invalid_load_state(runner, input_csv, tmp_path):
    """An edge case: invalid load-state JSON file (verify graceful error handling)."""
    invalid_state = tmp_path / "invalid.json"
    invalid_state.write_text("not a json")
    
    result = runner.invoke(main, ["--input", input_csv, "--load-state", str(invalid_state)], input="q\n")
    
    assert result.exit_code == 0
    assert f"Invalid state file {invalid_state}, starting fresh." in result.output

def test_quit_early(runner, input_csv):
    """An edge case: quitting early during an interactive session (e.g. sending 'q' input)."""
    result = runner.invoke(main, ["--input", input_csv], input="q\n")
    
    assert result.exit_code == 0
    assert "Quitting..." in result.output
    # It should still print final rankings if we quit? 
    # Looking at cli.py, if response == "q", it breaks the loop and proceeds to print rankings.
    assert "Item,Rank,Confidence,Uncertainty" in result.output
