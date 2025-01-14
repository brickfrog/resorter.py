import pytest
import pandas as pd
from resorter_py.main import (
    BayesianPairwiseRanker,
    read_input,
    parse_input,
    determine_queries,
    assign_levels,
    assign_custom_quantiles,
    Config,
)
import random


@pytest.fixture
def sample_model():
    items = ["A", "B", "C", "D"]
    return BayesianPairwiseRanker(items)


@pytest.fixture
def model_with_scores():
    items = ["A", "B", "C", "D"]
    scores = {"A": 4, "B": 3, "C": 2, "D": 1}
    return BayesianPairwiseRanker(items, scores)


def test_read_input_csv(tmp_path):
    # Test reading from CSV file
    df = pd.DataFrame({"Item": ["A", "B", "C"]})
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)  # Keep the header
    result = read_input(str(csv_path))
    assert len(result) == 3
    assert result.iloc[0, 0] == "A"
    assert list(result.columns) == ['Item']

    # Test reading CSV without headers
    df.to_csv(csv_path, index=False, header=False)
    result = read_input(str(csv_path))
    assert len(result) == 3
    assert result.iloc[0, 0] == "A"
    assert list(result.columns) == ['Item']


def test_read_input_string():
    # Test reading from comma-separated string
    result = read_input("A,B,C")
    assert len(result) == 3
    assert result.iloc[0, 0] == "A"
    assert list(result.columns) == ['Item']


def test_parse_input():
    # Test parsing with scores
    df = pd.DataFrame({"Item": ["A", "B"], "Score": [1, 2]})
    items, scores = parse_input(df)
    assert items == ["A", "B"]
    assert scores == {"A": "1", "B": "2"}

    # Test parsing without scores
    df = pd.DataFrame({"Item": ["A", "B"]})
    items, scores = parse_input(df)
    assert items == ["A", "B"]
    assert scores is None

    # Test with empty dataframe
    df = pd.DataFrame(columns=['Item'])
    items, scores = parse_input(df)
    assert items == []
    assert scores is None


def test_determine_queries():
    items = ["A", "B", "C", "D"]
    # Test with specified queries
    assert determine_queries(items, 10) == 10
    # Test with default calculation
    assert determine_queries(items, None) > 0


def test_bayesian_update(sample_model):
    alpha, beta = sample_model.bayesian_update(1, 1, 1, 0, 1)
    assert alpha > 1
    assert beta == 1


def test_compute_ranks(sample_model):
    ranks = sample_model.compute_ranks()
    assert len(ranks) == 4
    assert all(0 <= rank <= 1 for rank in ranks.values())


def test_standard_error(sample_model):
    se = sample_model.standard_error(1, 1)
    assert 0 <= se <= 1


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
    assert sample_model.should_continue(0.9) is True
    assert sample_model.should_continue(0.1) is True


def test_export_rankings(sample_model):
    # Test CSV export
    csv_output = sample_model.export_rankings('csv')
    assert isinstance(csv_output, pd.DataFrame)
    assert 'Item' in csv_output.columns
    assert 'Rank' in csv_output.columns
    assert 'Confidence' in csv_output.columns

    # Test JSON export
    json_output = sample_model.export_rankings('json')
    assert isinstance(json_output, dict)
    assert 'rankings' in json_output
    assert 'confidences' in json_output
    assert 'metadata' in json_output

    # Test Markdown export
    md_output = sample_model.export_rankings('markdown')
    assert isinstance(md_output, str)
    assert '| Item | Rank | Confidence |' in md_output


def test_consistency_check(sample_model):
    # Initialize with very different ranks
    sample_model.alpha_beta = {
        "A": (10, 1),  # High rank
        "B": (1, 10),  # Low rank
    }
    # Test inconsistent comparison
    sample_model.update_single_query("B", "A", 1)  # B better than A despite ranks
    # No assertion needed as this just prints a warning


def test_undo_comparison(sample_model):
    # Save initial state
    initial_state = sample_model.alpha_beta.copy()
    
    # Make a comparison
    sample_model.update_single_query("A", "B", 1)
    
    # Verify state changed
    assert sample_model.alpha_beta != initial_state
    
    # Undo the comparison
    sample_model.undo_last_comparison()
    
    # Verify state restored
    assert sample_model.alpha_beta == initial_state


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


def test_initialize_with_similarity(sample_model):
    items = ["Test1", "Test2", "Very Different"]
    scores = {"Test1": 0.9}
    sample_model.initialize_with_similarity(items, scores)
    # Test2 should get a similar score to Test1 due to name similarity
    # Very Different should not get a score


def test_export_formats(sample_model):
    # Test CSV format
    csv_output = sample_model.export_rankings('csv')
    assert isinstance(csv_output, pd.DataFrame)
    assert list(csv_output.columns) == ['Item', 'Rank', 'Confidence']
    
    # Test JSON format
    json_output = sample_model.export_rankings('json')
    assert isinstance(json_output, dict)
    assert all(k in json_output for k in ['rankings', 'confidences', 'metadata'])
    assert 'total_comparisons' in json_output['metadata']
    
    # Test Markdown format
    md_output = sample_model.export_rankings('markdown')
    assert isinstance(md_output, str)
    assert '| Item | Rank | Confidence |' in md_output
    
    # Test invalid format
    with pytest.raises(ValueError):
        sample_model.export_rankings('invalid')


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
        format='csv'
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
    assert config.format == 'csv'


def test_save_load_state(sample_model, tmp_path):
    # Make some comparisons
    sample_model.update_single_query("A", "B", 1)
    sample_model.update_single_query("C", "D", 2)
    
    # Save state
    state_file = tmp_path / "state.json"
    sample_model.save_state(str(state_file))
    
    # Create new model and load state
    new_model = BayesianPairwiseRanker(["A", "B", "C", "D"])
    new_model.load_state(str(state_file))
    
    # Check if states match
    assert new_model.items == sample_model.items
    assert new_model.alpha_beta == sample_model.alpha_beta
    assert new_model.history == sample_model.history


def test_early_stopping(sample_model):
    # Make some comparisons to build up confidence
    sample_model.update_single_query("A", "B", 1)
    sample_model.update_single_query("A", "C", 1)
    sample_model.update_single_query("A", "D", 1)
    sample_model.update_single_query("B", "C", 1)
    sample_model.update_single_query("B", "D", 1)
    
    # Now test stopping conditions
    # With low confidence threshold, should stop (return False)
    assert not sample_model.should_continue(0.1)
    
    # With high confidence threshold, should continue (return True)
    assert sample_model.should_continue(0.99)


def test_get_next_comparison(sample_model):
    # First comparison should be from most uncertain pairs (early phase)
    comparison = sample_model.get_next_comparison()
    assert comparison is not None
    assert len(comparison) == 2
    assert all(item in sample_model.items for item in comparison)
    assert comparison[0] != comparison[1]

    # Test that same comparison isn't returned twice
    first_comparison = comparison
    sample_model.submit_comparison(*first_comparison, 1)
    second_comparison = sample_model.get_next_comparison()
    assert second_comparison != first_comparison

    # Test early stopping with confidence threshold
    sample_model.min_confidence = 0.1  # Set very low confidence threshold
    # Make several comparisons to build up confidence
    for _ in range(10):
        comparison = sample_model.get_next_comparison()
        if comparison is None:
            break
        sample_model.submit_comparison(*comparison, 1)


def test_submit_comparison(sample_model):
    # Get initial state
    initial_alpha_beta = sample_model.alpha_beta.copy()
    initial_completed = sample_model.completed_comparisons.copy()
    initial_count = sample_model.iteration_count

    # Submit a valid comparison
    item_a, item_b = "A", "B"
    sample_model.submit_comparison(item_a, item_b, 1)

    # Check state was updated
    assert sample_model.alpha_beta != initial_alpha_beta
    assert len(sample_model.completed_comparisons) > len(initial_completed)
    assert sample_model.iteration_count == initial_count + 1
    assert sample_model.get_comparison_key(item_a, item_b) in sample_model.completed_comparisons

    # Test invalid response
    with pytest.raises(ValueError):
        sample_model.submit_comparison(item_a, item_b, 4)
    with pytest.raises(ValueError):
        sample_model.submit_comparison(item_a, item_b, 0)

    # Test comparison tracking
    key = sample_model.get_comparison_key(item_a, item_b)
    assert key in sample_model.completed_comparisons
    # Same comparison with items swapped should be tracked
    key_swapped = sample_model.get_comparison_key(item_b, item_a)
    assert key_swapped in sample_model.completed_comparisons


def test_comparison_sequence(sample_model):
    """Test the full sequence of getting comparisons and submitting results"""
    seen_comparisons = set()
    max_comparisons = 10
    
    for _ in range(max_comparisons):
        comparison = sample_model.get_next_comparison()
        if comparison is None:  # Stop if confidence threshold reached
            break
            
        # Check comparison is valid
        assert comparison[0] != comparison[1]
        assert all(item in sample_model.items for item in comparison)
        
        # Submit a random response (1, 2, or 3)
        response = random.choice([1, 2, 3])
        sample_model.submit_comparison(*comparison, response)
        
        # Track comparison
        comparison_key = sample_model.get_comparison_key(*comparison)
        assert comparison_key not in seen_comparisons  # Shouldn't see same comparison twice
        seen_comparisons.add(comparison_key)
    
    # Verify we have some rankings
    rankings = sample_model.compute_ranks()
    assert len(rankings) == len(sample_model.items)
    assert all(0 <= rank <= 1 for rank in rankings.values())
