from resorter_py.ranker import BradleyTerryRanker

def example_llm_response(item_a: str, item_b: str) -> int:
    """Simulate an LLM response for testing.
    In reality, this would call your actual LLM."""
    print(f"\nComparing: '{item_a}' vs '{item_b}'")
    while True:
        try:
            response = input("Enter 1 (first is better), 2 (equal), or 3 (second is better): ")
            if response in ["1", "2", "3"]:
                return int(response)
            print("Invalid input. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter 1, 2, or 3.")

def main():
    # Initialize the ranker with some test items
    items = ["The Matrix", "Inception", "Interstellar", "The Dark Knight"]
    ranker = BradleyTerryRanker(items)
    min_confidence = 0.8  # Target confidence threshold

    print("\nStarting comparison sequence...")
    print(f"We'll compare movies until we reach {min_confidence:.0%} confidence in the rankings.")
    
    # Get comparisons one at a time
    while ranker.should_continue(min_confidence=min_confidence):
        item_a, item_b = ranker.get_most_informative_pair()
        if item_a is None:  # All possible unique comparisons completed
            break
            
        # Get answer from simulated LLM
        response = example_llm_response(item_a, item_b)
        
        print(f"Response: {response}")

        # Submit the answer back to the ranker
        ranker.update_single_query(item_a, item_b, response)
        
        # Show current rankings after each comparison
        print("\nCurrent Rankings:")
        rankings = ranker.export_rankings(format='json')
        for item, rank in rankings['rankings'].items():
            confidence = rankings['confidences'][item]
            print(f"{item}: {rank:.3f} (confidence: {confidence:.1%})")

    # Get and display final rankings
    print("\nFinal Rankings:")
    final_rankings = ranker.export_rankings(format='json')
    for item, rank in final_rankings['rankings'].items():
        confidence = final_rankings['confidences'][item]
        print(f"{item}: {rank:.3f} (confidence: {confidence:.1%})")
    
    print(f"\nTotal comparisons made: {final_rankings['metadata']['total_comparisons']}")
    print(f"Mean uncertainty: {final_rankings['metadata']['mean_uncertainty']:.3f}")

if __name__ == "__main__":
    main()