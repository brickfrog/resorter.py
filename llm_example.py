from resorter_py.ranker import BradleyTerryRanker

def example_llm_response(item_a: str, item_b: str) -> int:
    """Simulate an LLM response for testing.
    In reality, this would call your actual LLM."""
    print(f"\nQuestion for LLM: Which movie is better: '{item_a}' or '{item_b}'?")
    while True:
        try:
            response = input("Enter 1 (first is better), 2 (equal), or 3 (second is better): ")
            if response in ["1", "2", "3"]:
                return int(response)
            print("Invalid input. Please enter 1, 2, or 3.")
        except (EOFError, KeyboardInterrupt):
            print("\nInput interrupted. Exiting.")
            raise SystemExit(1)


def main():
    # Initialize the ranker with some test items
    items = ["The Matrix", "Inception", "Interstellar", "The Dark Knight"]
    model = BradleyTerryRanker(items)

    print("\nStarting comparison sequence...")
    print("We'll compare movies until we reach sufficient confidence in the rankings.")
    
    # Get comparisons one at a time
    while model.should_continue(min_confidence=0.8):
        item_a, item_b = model.get_most_informative_pair()
        if item_a is None or item_b is None:  # Done when no more pairs
            break
            
        # Get answer from simulated LLM
        response = example_llm_response(item_a, item_b)
        
        print(f"Response: {response}")

        # Submit the answer back to the ranker
        model.update_single_query(item_a, item_b, response)

        # Show current rankings after each comparison
        print("\nCurrent Rankings:")
        ranks = model.compute_ranks()
        confidences = model.get_ranking_confidence()

        for item, rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
            confidence = confidences[item]
            print(f"{item}: {rank:.3f} (confidence: {confidence:.1%})")

    # Get and display final rankings using export_rankings(format='json')
    print("\nFinal Rankings (from JSON export):")
    final_rankings = model.export_rankings(format='json')
    for item, rank in final_rankings['rankings'].items():
        confidence = final_rankings['confidences'][item]
        print(f"{item}: {rank:.3f} (confidence: {confidence:.1%})")
    
    print(f"\nTotal comparisons made: {final_rankings['metadata']['total_comparisons']}")
    print(f"Mean uncertainty: {final_rankings['metadata']['mean_uncertainty']:.3f}")

if __name__ == "__main__":
    main()
