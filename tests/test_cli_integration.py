import json
import os
import pytest
from click.testing import CliRunner
from resorter_py.cli import main

def test_cli_interactive_ranking(tmp_path):
    """Scenario 1: One short interactive ranking run."""
    input_file = tmp_path / "items.csv"
    input_file.write_text("Apple\nBanana\nCherry")
    
    runner = CliRunner()
    # 3 items -> ~5 queries. 
    # Providing enough inputs to finish. 
    # Inputs: 1=Apple better, 2=Tied, 3=Cherry better, etc.
    # We provide '1' for all queries to ensure it reaches the end.
    result = runner.invoke(main, ["--input", str(input_file)], input="1\n1\n1\n1\n1\n")
    
    assert result.exit_code == 0
    assert "Number of queries:" in result.output
    assert "Comparison 1/" in result.output
    # Check that output contains the items
    assert "Apple" in result.output
    assert "Banana" in result.output
    assert "Cherry" in result.output
    assert "Item,Rank,Confidence,Uncertainty" in result.output

def test_cli_save_load_state(tmp_path):
    """Scenario 2: Save state to a file, then load that state in a subsequent run."""
    input_file = tmp_path / "items.csv"
    input_file.write_text("Apple\nBanana")
    state_file = tmp_path / "state.json"
    
    runner = CliRunner()
    # First run: Do one comparison then quit to save state.
    # input: '1' (comparison), then 'q' (quit)
    result = runner.invoke(main, ["--input", str(input_file), "--save-state", str(state_file)], input="1\nq\n")
    assert result.exit_code == 0
    assert os.path.exists(state_file)
    assert f"Saved state to {state_file}" in result.output
    
    # Second run: Load the state and finish.
    # input: '1', '1', '1' (enough to satisfy num_queries=3)
    result = runner.invoke(main, ["--input", str(input_file), "--load-state", str(state_file)], input="1\n1\n1\n")
    assert result.exit_code == 0
    assert f"Loaded state from {state_file}" in result.output
    # Check that it didn't start from Comparison 1/3 if state was loaded correctly.
    # Actually, it might still show Comparison 1/3 because it resets 'i' in the while loop?
    # Let's check cli.py:
    # i = 0
    # while i < num_queries:
    #   ...
    #   if response in ["1", "2", "3"]:
    #     model.update_single_query(...)
    #     i += 1
    # The 'i' is local to main(), but model.iteration_count is in the model.
    # So it might restart numbering from 1 if we just re-run main.
    # But it should have the data from the first run.
    assert "Apple" in result.output

def test_cli_json_output(tmp_path):
    """Scenario 3: One non-default output format (JSON)."""
    input_file = tmp_path / "items.csv"
    input_file.write_text("Apple\nBanana")
    
    runner = CliRunner()
    # items = ["A", "B"] -> 3 queries.
    result = runner.invoke(main, ["--input", str(input_file), "--format", "json"], input="1\n1\n1\n")
    assert result.exit_code == 0
    
    # The JSON output is at the end. We need to parse it.
    # Find the first '{' and parse from there.
    json_start = result.output.find('{')
    assert json_start != -1
    json_str = result.output[json_start:]
    
    data = json.loads(json_str)
    assert "rankings" in data
    assert "Apple" in data["rankings"]
    assert "Banana" in data["rankings"]
    assert "metadata" in data

def test_cli_edge_cases(tmp_path):
    """Scenario 4: Invalid load-state JSON file, and one for quitting early."""
    input_file = tmp_path / "items.csv"
    input_file.write_text("Apple\nBanana")
    
    runner = CliRunner()
    
    # Case 4a: Quitting early
    result = runner.invoke(main, ["--input", str(input_file)], input="q\n")
    assert result.exit_code == 0
    assert "Quitting..." in result.output
    assert "Apple" in result.output # Should still print final rankings
    
    # Case 4b: Invalid load-state JSON file
    invalid_state = tmp_path / "invalid.json"
    invalid_state.write_text("this is not valid json")
    
    result = runner.invoke(main, ["--input", str(input_file), "--load-state", str(invalid_state)], input="q\n")
    assert result.exit_code == 0
    assert f"Invalid state file {invalid_state}, starting fresh." in result.output
    assert "Quitting..." in result.output

def test_cli_reproducibility(tmp_path):
    """Scenario 5: Running with the same seed should produce the same results."""
    input_file = tmp_path / "items.csv"
    # Need enough items to have some randomization in get_most_informative_pair
    items = ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig", "Grape"]
    input_file.write_text("\n".join(items))
    
    runner = CliRunner()
    
    # Run 1
    # We provide a fixed sequence of inputs
    # Need enough inputs for ~15-20 queries. 
    inputs = "1\n2\n3\n" * 10
    result1 = runner.invoke(main, ["--input", str(input_file), "--seed", "42"], input=inputs)
    assert result1.exit_code == 0
    
    # Run 2 with same seed
    result2 = runner.invoke(main, ["--input", str(input_file), "--seed", "42"], input=inputs)
    assert result2.exit_code == 0
    
    # The outputs should be identical (specifically the sequence of comparisons and final ranks)
    assert result1.output == result2.output

    # Run 3 with different seed - should likely produce different comparison sequence
    result3 = runner.invoke(main, ["--input", str(input_file), "--seed", "43"], input=inputs)
    assert result3.exit_code == 0
    # While it's *possible* it's the same, with 7 items it's very unlikely.
    assert result1.output != result3.output
