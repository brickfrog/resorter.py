import importlib.util
import os


def test_llm_example_importable():
    """Test that the llm_example script can be imported without error."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    llm_example_path = os.path.join(root_dir, "llm_example.py")

    try:
        spec = importlib.util.spec_from_file_location("llm_example", llm_example_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for llm_example at {llm_example_path}")
        llm_example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llm_example)
        assert llm_example.BradleyTerryRanker is not None
        assert callable(llm_example.main)
    except ImportError as e:
        # Check if it was some other import error within the file
        raise ImportError(f"Failed to import llm_example: {e}")
