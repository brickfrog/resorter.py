import pytest
import pandas as pd
import numpy as np
from resorter_py.ranker import (
    BradleyTerryRanker,
    read_input,
    parse_input,
    determine_queries,
    assign_levels,
    assign_custom_quantiles,
)
from resorter_py.cli import Config
import random


@pytest.fixture
def sample_model():
    items = ["A", "B", "C", "D"]
    return BradleyTerryRanker(items)


@pytest.fixture
def model_with_scores():
    items = ["A", "B", "C", "D"]
    scores = {"A": 4, "B": 3, "C": 2, "D": 1}
    return BradleyTerryRanker(items, scores)


@pytest.fixture
def bt_sample_model():
    items = ["A", "B", "C", "D"]
    return BradleyTerryRanker(items)


@pytest.fixture
def bt_model_with_scores():
    items = ["A", "B", "C", "D"]
    scores = {"A": 4, "B": 3, "C": 2, "D": 1}
    return BradleyTerryRanker(items, scores)


def test_read_input_csv(tmp_path):
    # Test reading from CSV file
    df = pd.DataFrame({"Item": ["A", "B", "C"]})
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)  # Keep the header
    result = read_input(str(csv_path))
    assert len(result) == 3
    assert result.iloc[0, 0] == "A"
    assert list(result.columns) == ["Item"]

    # Test reading CSV without headers
    df.to_csv(csv_path, index=False, header=False)
    result = read_input(str(csv_path))
    assert len(result) == 3
    assert result.iloc[0, 0] == "A"
    assert list(result.columns) == ["Item"]


def test_read_input_string():
    # Test reading from comma-separated string
    result = read_input("A,B,C")
    assert len(result) == 3
    assert result.iloc[0, 0] == "A"
    assert list(result.columns) == ["Item"]


def test_parse_input():
    # Test parsing with scores
    df = pd.DataFrame({"Item": ["A", "B"], "Score": [1, 2]})
    items, scores = parse_input(df)
    assert items == ["A", "B"]
    assert scores == {"A": 1.0, "B": 2.0}

    # Test parsing without scores
    df = pd.DataFrame({"Item": ["A", "B"]})
    items, scores = parse_input(df)
    assert items == ["A", "B"]
    assert scores is None

    # Test with empty dataframe
    df = pd.DataFrame(columns=["Item"])
    items, scores = parse_input(df)
    assert items == []
    assert scores is None


def test_determine_queries():
    items = ["A", "B", "C", "D"]
    # Test with specified queries
    assert determine_queries(items, 10) == 10
    # Test with default calculation
    assert determine_queries(items, None) > 0


def test_compute_ranks(sample_model):
    ranks = sample_model.compute_ranks()
    assert len(ranks) == 4
    assert all(0 <= rank <= 1 for rank in ranks.values())


def test_get_mean_uncertainty(sample_model):
    uncertainty = sample_model.get_mean_uncertainty()
    assert 0 <= uncertainty <= 1


def test_get_ranking_confidence(sample_model):
    confidences = sample_model.get_ranking_confidence()
    assert len(confidences) == 4
    assert all(0 <= conf <= 1 for conf in confidences.values())


def test_get_comparison_key(sample_model):
    # Test that keys are consistent regardless of order
    key1 = sample_model.get_comparison_key("A", "B")
    key2 = sample_model.get_comparison_key("B", "A")
    assert key1 == key2


def test_should_continue(sample_model):
    # Test early stopping condition
    assert sample_model.should_continue(0.9) is True  # Initially uncertain
    
    # Make several comparisons to build confidence
    for _ in range(10):
        item_a, item_b = sample_model.get_most_informative_pair()
        sample_model.update_single_query(item_a, item_b, 1)
    
    # Now should be more confident
    assert not sample_model.should_continue(0.1)


def test_export_rankings(sample_model):
    # Test CSV export
    csv_output = sample_model.export_rankings("csv")
    assert isinstance(csv_output, pd.DataFrame)
    assert "Item" in csv_output.columns
    assert "Rank" in csv_output.columns
    assert "Confidence" in csv_output.columns

    # Test JSON export
    json_output = sample_model.export_rankings("json")
    assert isinstance(json_output, dict)
    assert "rankings" in json_output
    assert "confidences" in json_output
    assert "metadata" in json_output

    # Test Markdown export
    md_output = sample_model.export_rankings("markdown")
    assert isinstance(md_output, str)
    assert "| Item | Rank | Confidence |" in md_output


def test_consistency_check(sample_model):
    # Initialize with very different strengths
    sample_model.strengths[0] = 2.0  # High strength for A
    sample_model.strengths[1] = -2.0  # Low strength for B
    
    # Test inconsistent comparison
    sample_model.update_single_query("B", "A", 1)  # B better than A despite strengths
    # No assertion needed as this just prints a warning


def test_undo_comparison(sample_model):
    # Save initial state
    initial_strengths = sample_model.strengths.copy()
    initial_comparisons = sample_model.comparison_matrix.copy()
    initial_wins = sample_model.win_matrix.copy()

    # Make a comparison
    sample_model.update_single_query("A", "B", 1)

    # Verify state changed
    assert not np.array_equal(sample_model.strengths, initial_strengths)

    # Undo the comparison
    sample_model.undo_last_comparison()

    # Verify state restored
    assert np.array_equal(sample_model.strengths, initial_strengths)
    assert np.array_equal(sample_model.comparison_matrix, initial_comparisons)
    assert np.array_equal(sample_model.win_matrix, initial_wins)


def test_assign_levels():
    sorted_ranks = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3}
    levels = assign_levels(sorted_ranks, 2)
    assert len(levels) == 4
    assert all(level in [1, 2] for level in levels.values())


def test_assign_custom_quantiles():
    sorted_ranks = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3}
    quantiles = assign_custom_quantiles(sorted_ranks, [0, 0.5, 1])
    assert len(quantiles) == 4
    assert all(quantile in [1, 2, 3] for quantile in quantiles.values())


def test_config():
    config = Config(
        input="test.csv",
        output="out.csv",
        queries=10,
        levels=5,
        quantiles="0 0.5 1",
        progress=True,
        save_state="state.json",
        load_state=None,
        min_confidence=0.9,
        visualize=True,
        format="csv",
    )
    assert config.input == "test.csv"
    assert config.output == "out.csv"
    assert config.queries == 10
    assert config.levels == 5
    assert config.quantiles == "0 0.5 1"
    assert config.progress is True
    assert config.save_state == "state.json"
    assert config.load_state is None
    assert config.min_confidence == 0.9
    assert config.visualize is True
    assert config.format == "csv"


def test_save_load_state(sample_model, tmp_path):
    # Make some comparisons
    sample_model.update_single_query("A", "B", 1)
    sample_model.update_single_query("C", "D", 2)
    
    # Record state
    state_iteration_count = sample_model.iteration_count
    state_strengths = sample_model.strengths.copy()
    state_comparisons = sample_model.comparison_matrix.copy()
    state_wins = sample_model.win_matrix.copy()

    # Save state
    state_file = tmp_path / "state.json"
    sample_model.save_state(str(state_file))

    # Create new model and load state
    new_model = BradleyTerryRanker(["A", "B", "C", "D"])
    new_model.load_state(str(state_file))

    # Check if states match
    assert new_model.items == sample_model.items
    assert np.array_equal(new_model.strengths, state_strengths)
    assert np.array_equal(new_model.comparison_matrix, state_comparisons)
    assert np.array_equal(new_model.win_matrix, state_wins)
    assert new_model.iteration_count == state_iteration_count


def test_comparison_sequence(sample_model):
    """Test the full sequence of getting comparisons and submitting results"""
    seen_comparisons = set()
    max_comparisons = 10
    total_possible_pairs = (len(sample_model.items) * (len(sample_model.items) - 1)) // 2

    for _ in range(max_comparisons):
        # Get next comparison using most informative pair
        item_a, item_b = sample_model.get_most_informative_pair()
        if item_a is None or item_b is None:  # Stop if no more pairs
            break

        # Check comparison is valid
        assert item_a != item_b
        assert all(item in sample_model.items for item in [item_a, item_b])

        # Submit a random response (1, 2, or 3)
        response = random.choice([1, 2, 3])
        sample_model.update_single_query(item_a, item_b, response)

        # Track comparison
        comparison_key = sample_model.get_comparison_key(item_a, item_b)
        seen_comparisons.add(comparison_key)

    # Verify we have some rankings and they make sense
    rankings = sample_model.compute_ranks()
    assert len(rankings) == len(sample_model.items)
    assert all(0 <= rank <= 1 for rank in rankings.values())
    # Verify we've made some comparisons
    assert len(seen_comparisons) > 0
    # Verify we haven't exceeded possible pairs
    assert len(seen_comparisons) <= total_possible_pairs


def test_bradley_terry_prob(sample_model):
    prob = sample_model.bradley_terry_prob(1.0, 0.0)
    assert 0 < prob < 1
    assert abs(sample_model.bradley_terry_prob(0.0, 0.0) - 0.5) < 1e-6


def test_bradley_terry_update(sample_model):
    # Initial state
    initial_strengths = sample_model.strengths.copy()
    
    # Make a comparison
    sample_model.update_single_query("A", "B", 1)  # A wins
    
    # Check that strengths were updated
    assert not np.array_equal(sample_model.strengths, initial_strengths)
    assert sample_model.comparison_matrix[0, 1] == 1
    assert sample_model.win_matrix[0, 1] == 1


def test_bradley_terry_compute_ranks(sample_model):
    # Make some comparisons
    sample_model.update_single_query("A", "B", 1)  # A > B
    sample_model.update_single_query("B", "C", 1)  # B > C
    sample_model.update_single_query("C", "D", 1)  # C > D
    
    # Get rankings
    ranks = sample_model.compute_ranks()
    
    # Check that rankings make sense
    assert ranks["A"] > ranks["B"]
    assert ranks["B"] > ranks["C"]
    assert ranks["C"] > ranks["D"]


def test_bradley_terry_uncertainty(sample_model):
    # Initial uncertainty should be high
    initial_uncertainty = sample_model.get_uncertainty("A")
    assert initial_uncertainty == 1.0
    
    # Make several comparisons to build confidence
    for _ in range(5):
        item_a, item_b = sample_model.get_most_informative_pair()
        sample_model.update_single_query(item_a, item_b, 1)
    
    # Uncertainty should decrease
    new_uncertainty = sample_model.get_uncertainty("A")
    assert new_uncertainty < initial_uncertainty


def test_bradley_terry_informative_pair(sample_model):
    # Initial pair should be from items with similar strengths
    item_a, item_b = sample_model.get_most_informative_pair()
    assert item_a in sample_model.items
    assert item_b in sample_model.items
    assert item_a != item_b
    
    # Make several comparisons to change uncertainties
    for _ in range(3):
        sample_model.update_single_query("A", "B", 1)
        sample_model.update_single_query("A", "C", 1)
        sample_model.update_single_query("B", "C", 1)
    
    # Get new pair and verify it's valid
    new_a, new_b = sample_model.get_most_informative_pair()
    assert new_a is not None and new_b is not None
    assert new_a != new_b
    assert new_a in sample_model.items
    assert new_b in sample_model.items
    
    # Verify the pair selection makes sense
    uncertainty_a = sample_model.get_uncertainty(new_a)
    uncertainty_b = sample_model.get_uncertainty(new_b)
    # At least one of the items should have high uncertainty
    assert max(uncertainty_a, uncertainty_b) > 0.5
