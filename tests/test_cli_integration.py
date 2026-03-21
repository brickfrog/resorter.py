import json
import pytest
from click.testing import CliRunner
from resorter_py.cli import main

@pytest.fixture
def input_csv(tmp_path):
    p = tmp_path / "items.csv"
    p.write_text("A\nB\nC\n")
    return str(p)

def test_cli_interactive_ranking(input_csv):
    """Test a short interactive ranking run."""
    runner = CliRunner()
    # Simulate user input: 
    # 1 for first comparison, 
    # 3 for second, 
    # 2 for third
    # This should be enough to finish if we set queries low
    result = runner.invoke(main, ["--input", input_csv, "--queries", "3"], input="1\n3\n2\n")
    
    assert result.exit_code == 0
    assert "Number of queries: 3" in result.output
    assert "Comparison 1/3" in result.output
    assert "Comparison 2/3" in result.output
    assert "Comparison 3/3" in result.output
    assert "Item,Rank,Confidence,Uncertainty" in result.output

def test_cli_save_load_state(input_csv, tmp_path):
    """Test saving state to a file and then loading it."""
    state_file = tmp_path / "state.json"
    runner = CliRunner()
    
    # Run 1: Save state after 1 comparison
    result1 = runner.invoke(
        main, 
        ["--input", input_csv, "--queries", "2", "--save-state", str(state_file)], 
        input="1\nq\n"
    )
    assert result1.exit_code == 0
    assert f"Saved state to {state_file}" in result1.output
    assert state_file.exists()
    
    # Run 2: Load state and continue
    result2 = runner.invoke(
        main, 
        ["--input", input_csv, "--queries", "2", "--load-state", str(state_file)], 
        input="3\nq\n"
    )
    if result2.exit_code != 0:
        print(f"Result 2 output: {result2.output}")
        if result2.exception:
            print(f"Result 2 exception: {result2.exception}")
    assert result2.exit_code == 0
    assert f"Loaded state from {state_file}" in result2.output
    # It should show Comparison 1/2 because i starts at 0 in cli.py every run
    assert "Comparison 1/2" in result2.output

def test_cli_json_format(input_csv):
    """Test non-default output format (JSON)."""
    runner = CliRunner()
    # Using --queries 1 to make it quick
    result = runner.invoke(main, ["--input", input_csv, "--queries", "1", "--format", "json"], input="1\n")
    
    assert result.exit_code == 0
    
    # Search for the JSON object in the output
    try:
        start_idx = result.output.find('{')
        json_str = result.output[start_idx:]
        data = json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        pytest.fail(f"Failed to parse JSON from output: {result.output}\nError: {e}")
    
    assert "rankings" in data
    assert "confidences" in data
    assert "uncertainties" in data
    assert "metadata" in data

def test_cli_edge_case_invalid_state(input_csv, tmp_path):
    """Test loading an invalid JSON state file."""
    invalid_state = tmp_path / "invalid.json"
    invalid_state.write_text("not a json")
    
    runner = CliRunner()
    result = runner.invoke(
        main, 
        ["--input", input_csv, "--queries", "1", "--load-state", str(invalid_state)], 
        input="1\n"
    )
    
    assert result.exit_code == 0
    assert f"Invalid state file {invalid_state}, starting fresh." in result.output
    assert "Comparison 1/1" in result.output

def test_cli_edge_case_quit_early(input_csv):
    """Test quitting early by entering 'q'."""
    runner = CliRunner()
    result = runner.invoke(
        main, 
        ["--input", input_csv, "--queries", "5"], 
        input="q\n"
    )
    
    assert result.exit_code == 0
    assert "Quitting..." in result.output
    # It should still print final rankings
    assert "Item,Rank,Confidence,Uncertainty" in result.output
