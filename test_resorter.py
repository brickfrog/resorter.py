import subprocess
import csv
import pandas as pd
from resorter import (
    parse_input,
    determine_queries,
    generate_bin_edges,
    BradleyTerryModel,
)
from unittest.mock import patch


class TestCodeUnderTest:
    # Test parsing input with two columns
    def test_parsing_input_with_two_columns(self):
        # Given
        df = pd.DataFrame([["item1", 10], ["item2", 20]])

        # When
        items, scores = parse_input(df)

        # Then
        assert items == ["item1", "item2"]
        assert scores == {"item1": 10, "item2": 20}

    # Test parsing input with one column
    def test_parsing_input_with_one_column(self):
        # Given
        df = pd.DataFrame(["item1", "item2"], columns=["Item"])

        # When
        items, scores = parse_input(df)

        # Then
        assert items == ["item1", "item2"]
        assert scores is None

    # Test determining number of queries with user input
    def test_determining_number_of_queries_with_user_input(self):
        # Given
        items = ["item1", "item2", "item3"]
        args_queries = 5

        # When
        queries = determine_queries(items, args_queries)

        # Then
        assert queries == 5

    # Test parsing empty input
    def test_parsing_empty_input(self):
        # Given
        df = pd.DataFrame([], columns=["Item"])

        # When
        items, scores = parse_input(df)

        # Then
        assert items == []
        assert scores is None

    # Test determining number of queries with empty input
    def test_determining_number_of_queries_with_empty_input(self):
        # Given
        items = []
        args_queries = None

        # When
        queries = determine_queries(items, args_queries)

        # Then
        assert queries == 0

    # Test generating bin edges with empty data
    def test_generating_bin_edges_with_empty_data(self):
        # Given
        data = []

        # When
        bin_edges = generate_bin_edges(data)

        # Then
        assert bin_edges is None

    # Test determining number of queries without user input
    def test_determining_number_of_queries_without_user_input(self):
        # Given
        items = ["item1", "item2", "item3", "item4"]

        # When
        num_queries = determine_queries(items, None)

        # Then
        # should be using int(ceil(len(items) * np.log(len(items)) + 1))
        assert num_queries == 7

    # Test generating bin edges with levels
    def test_generating_bin_edges_with_levels(self):
        # Given
        data = [1, 2, 3, 4, 5]
        levels = 3

        # When
        bin_edges = generate_bin_edges(data, levels=levels)

        # Then
        assert bin_edges is not None
        assert len(bin_edges) == levels + 1
        assert bin_edges[0] == 1
        assert bin_edges[-1] == 5

    # Test generating bin edges with quantiles
    def test_generating_bin_edges_with_quantiles(self):
        # Given
        data = [1, 2, 3, 4, 5]
        quantiles = 4

        # When
        bin_edges = generate_bin_edges(data, quantiles=quantiles)

        # Then
        assert bin_edges is not None
        assert len(bin_edges) == quantiles + 1
        assert bin_edges[0] == 1
        assert bin_edges[-1] == 5

    def test_standard_error_after_random_choice(mocker):
        # Given
        model = BradleyTerryModel(["item1", "item2", "item3"])

        with patch("builtins.input", return_value="2"):
            comparison_data = model.generate_comparison_data(1)

        # When
        item = comparison_data[0][0]  # get the first item from the first comparison
        alpha, beta = model.alpha_beta[item]
        computed_se = model.standard_error(alpha, beta)
        expected_se = (
            (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        ) ** 0.5

        # Then
        assert computed_se == expected_se

    def test_cli_integration(self):
        # Set up test input file
        with open("input.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Inky"])
            writer.writerow(["Pinky"])
            writer.writerow(["Blinky"])

        # Simulate the user input for the choices
        user_input = "1\n1\n1\n1\n1"

        # Call the CLI script and provide the simulated user input
        result = subprocess.run(
            ["python", "resorter.py", "--input", "input.csv", "--output", "output.csv"],
            input=user_input,
            capture_output=True,
            text=True,
        )

        # Check if the script ran successfully
        assert result.returncode == 0

        # (Optional) Check contents of out.csv or other expected behaviors
        with open("output.csv", "r", newline="") as csvfile:
            csv.reader(csvfile)
            # reader = csv.reader(csvfile)
            # Perform assertions based on the expected output in rows
            # Future TODO: assertions on input / need to control random choice

        # Cleanup
        subprocess.run(["rm", "input.csv"])
        subprocess.run(["rm", "output.csv"])
