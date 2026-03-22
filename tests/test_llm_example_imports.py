import pytest

def test_llm_example_imports():
    """Verify that the imports and methods used in llm_example.py work as expected."""
    # This mirrors the imports in llm_example.py
    from resorter_py.ranker import BradleyTerryRanker
    
    items = ["A", "B", "C"]
    model = BradleyTerryRanker(items)
    
    # Verify methods used in llm_example.py exist and are callable
    assert callable(model.should_continue)
    assert callable(model.get_most_informative_pair)
    assert callable(model.update_single_query)
    assert callable(model.compute_ranks)
    assert callable(model.get_ranking_confidence)
    assert callable(model.export_rankings)
    
    # Basic functional test of methods to ensure they don't crash
    assert model.should_continue(0.8) is True
    pair = model.get_most_informative_pair()
    assert pair[0] in items and pair[1] in items
    
    # Update with a dummy response
    model.update_single_query(pair[0], pair[1], 1)
    
    ranks = model.compute_ranks()
    assert len(ranks) == 3
    
    confidences = model.get_ranking_confidence()
    assert len(confidences) == 3
    
    export = model.export_rankings(format='json')
    assert 'rankings' in export
    assert 'confidences' in export
    assert 'metadata' in export
