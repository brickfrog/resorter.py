import importlib.util
import os
import sys

def test_llm_example_importable():
    """Test that the llm_example script can be imported without error."""
    # Add root to sys.path if not already there
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        
    try:
        import llm_example
        assert llm_example.BradleyTerryRanker is not None
        assert callable(llm_example.main)
    except ImportError as e:
        # Check if it was some other import error within the file
        raise ImportError(f"Failed to import llm_example: {e}")
