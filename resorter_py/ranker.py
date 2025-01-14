import random
from math import ceil, sqrt
from typing import Dict, List, Optional, Tuple, Union
import os
from difflib import SequenceMatcher
import json

import numpy as np
import pandas as pd


def read_input(data_input: str) -> pd.DataFrame:
    """
    Reads .csv into a dataframe, or splits a comma-separated string into a dataframe.
    Returns DataFrame with 'Item' and optionally 'Score' columns.
    """
    if os.path.exists(data_input) and data_input.endswith(".csv"):
        try:
            # First try reading with headers
            df = pd.read_csv(data_input, dtype=str, na_filter=False)
            if "Item" not in df.columns:
                # If headers don't match, read without headers and set them
                df = pd.read_csv(data_input, header=None, dtype=str, na_filter=False)
                if df.shape[1] == 2:
                    df.columns = ["Item", "Score"]
                else:
                    df.columns = ["Item"]
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["Item"])
        return df

    # Handle comma-separated string input
    items = [item.strip() for item in data_input.split(",")]
    return pd.DataFrame({"Item": items})


def parse_input(
    df: pd.DataFrame,
) -> Tuple[List[Union[int, str]], Optional[Dict[Union[int, str], float]]]:
    """
    Parse the input dataframe to separate items and scores.
    Expects DataFrame with 'Item' and optionally 'Score' columns.
    """
    items = df["Item"].tolist()
    if "Score" in df.columns:
        scores = df.set_index("Item")["Score"].astype(str).to_dict()
    else:
        scores = None
    return items, scores


def determine_queries(items: List[Union[int, str]], args_queries: Optional[int]) -> int:
    """
    Determine the number of queries based on the list length and user input.
    """
    if args_queries is not None:
        return args_queries

    list_length = len(items)
    if list_length == 0:
        return 0

    return int(ceil(list_length * np.log(list_length) + 1))


def generate_bin_edges(
    data: List[float], levels: Optional[int] = None, quantiles: Optional[int] = None
) -> Optional[np.ndarray]:
    if quantiles is not None:
        return np.quantile(data, np.linspace(0, 1, quantiles + 1))
    elif levels is not None:
        return np.linspace(min(data), max(data), levels + 1)
    else:
        return None


def assign_custom_quantiles(
    sorted_ranks: Dict[Union[int, str], float], quantile_cutoffs: List[float]
) -> Dict[Union[int, str], int]:
    sorted_values = [val for _, val in sorted_ranks.items()]
    num_items = len(sorted_values)

    cutoff_positions = [int(c * num_items) for c in quantile_cutoffs]

    quantiles = {}
    quantile_label = 0

    for i, (key, _) in enumerate(sorted_ranks.items()):
        if (
            quantile_label < len(cutoff_positions) - 1
            and i >= cutoff_positions[quantile_label]
        ):
            quantile_label += 1
        quantiles[key] = quantile_label + 1

    return quantiles


def assign_levels(
    sorted_ranks: Dict[Union[int, str], float], num_levels: int
) -> Dict[Union[int, str], int]:
    total_items = len(sorted_ranks)
    items_per_level = total_items // num_levels
    remainder = total_items % num_levels
    levels = {}
    current_level = num_levels
    count = 0
    for key, _ in sorted_ranks.items():
        levels[key] = current_level
        count += 1
        if count >= items_per_level:
            if remainder > 0:
                remainder -= 1
            else:
                current_level -= 1
                count = 0
    return levels


class BayesianPairwiseRanker:
    def __init__(
        self,
        items: List[Union[int, str]],
        scores: Optional[Dict[Union[int, str], float]] = None,
    ) -> None:
        self.items: List[Union[int, str]] = items
        self.alpha_beta: Dict[Union[int, str], Tuple[float, float]]
        self.iteration_count: int = 0
        self.completed_comparisons: set = set()
        if scores:
            self.alpha_beta = {
                item: (float(score), 1) for item, score in scores.items()
            }
        else:
            self.alpha_beta = {item: (1, 1) for item in items}
        self.history: List[
            Tuple[str, str, Dict[Union[int, str], Tuple[float, float]]]
        ] = []

    @staticmethod
    def standard_error(alpha: float, beta: float) -> float:
        return sqrt((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1)))

    def bayesian_update(
        self, alpha: float, beta: float, win: float, lose: float, iteration: int
    ) -> Tuple[float, float]:
        """Adaptive learning rate that decreases as we get more confident"""
        learning_rate = 1.0 / (1.0 + iteration * 0.1)  # Decreases over time
        return alpha + win * learning_rate, beta + lose * learning_rate

    def ask_question(
        self, item_a: Union[int, str], item_b: Union[int, str]
    ) -> Union[int, str]:
        while True:
            try:
                response = input(f"Is '{item_a}' better than '{item_b}'? ")
                if response in ["1", "2", "3"]:
                    self.update_single_query(item_a, item_b, int(response))
                    return int(response)
                elif response == "s":
                    print("Skipping...")
                    return "skip"
                elif response == "p":
                    self.print_estimates()
                elif response == "u":
                    self.undo_last_comparison()
                elif response == "q":
                    print("Quitting...")
                    exit(0)
                else:
                    print("Invalid input. Please enter 1, 2, 3, s, p, u, or q.")
            except ValueError:
                print("Invalid input. Please enter 1, 2, 3, s, p, u, or q.")

    def update_single_query(
        self, item_a: Union[int, str], item_b: Union[int, str], response: int
    ) -> None:
        # Save state before update
        self.history.append((item_a, item_b, self.alpha_beta.copy()))

        # Check consistency before updating
        rank_a = self.alpha_beta[item_a][0] / sum(self.alpha_beta[item_a])
        rank_b = self.alpha_beta[item_b][0] / sum(self.alpha_beta[item_b])
        rank_diff = abs(rank_a - rank_b)

        if rank_diff > 0.7:  # Large difference in current rankings
            if (response == 1 and rank_a < rank_b) or (
                response == 3 and rank_a > rank_b
            ):
                print(
                    "\nWarning: This comparison seems inconsistent with previous rankings!"
                )
                print("You might want to undo (u) and reconsider.")

        winners = [(1, 0), (0.5, 0.5), (0, 1)]
        win_a, win_b = winners[response - 1]

        self.iteration_count += 1
        for item, win, lose in [(item_a, win_a, win_b), (item_b, win_b, win_a)]:
            alpha, beta = self.alpha_beta[item]
            self.alpha_beta[item] = self.bayesian_update(
                alpha, beta, win, lose, self.iteration_count
            )

    def undo_last_comparison(self) -> None:
        """Undo the last comparison by restoring the previous state"""
        if not self.history:
            print("Nothing to undo!")
            return

        item_a, item_b, previous_state = self.history.pop()
        self.alpha_beta = previous_state
        print(f"Undid comparison between '{item_a}' and '{item_b}'")

    def print_estimates(self) -> None:
        mean_uncertainty = self.get_mean_uncertainty()
        confidences = self.get_ranking_confidence()

        print(f"\nMean uncertainty: {mean_uncertainty:.4f}")
        print("\nItem rankings:")
        sorted_items = sorted(
            self.alpha_beta.items(),
            key=lambda x: x[1][0] / (x[1][0] + x[1][1]),
            reverse=True,
        )
        for item, (alpha, beta) in sorted_items:
            rank = alpha / (alpha + beta)
            se = self.standard_error(alpha, beta)
            confidence = confidences[item]
            print(
                f"{item}: rank = {rank:.2f}, σ = {se:.4f}, confidence = {confidence:.2%}"
            )

    def get_most_informative_pair(self) -> Tuple[Union[int, str], Union[int, str]]:
        """Get the pair of items that would provide the most information."""
        uncertainties = {
            item: self.standard_error(alpha, beta)
            for item, (alpha, beta) in self.alpha_beta.items()
        }

        # Calculate expected information gain for each possible pair
        pairs = []
        for item_a in self.items:
            for item_b in self.items:
                if item_a >= item_b:  # Skip self-comparisons and duplicates
                    continue

                # Skip if we've already compared this pair
                if (
                    self.get_comparison_key(item_a, item_b)
                    in self.completed_comparisons
                ):
                    continue

                # Consider both uncertainty and how close the items are in ranking
                rank_a = self.alpha_beta[item_a][0] / sum(self.alpha_beta[item_a])
                rank_b = self.alpha_beta[item_b][0] / sum(self.alpha_beta[item_b])
                rank_diff = abs(rank_a - rank_b)

                # Items with similar ranks but high uncertainty are most informative
                information_value = (uncertainties[item_a] + uncertainties[item_b]) * (
                    1 - rank_diff
                )
                pairs.append((item_a, item_b, information_value))

        if not pairs:  # If all pairs have been compared
            self.completed_comparisons.clear()  # Reset completed comparisons
            return random.sample(self.items, 2)  # Return a random pair

        # Return the most informative pair
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[0][0], pairs[0][1]

    def get_comparison_key(
        self, item_a: Union[int, str], item_b: Union[int, str]
    ) -> tuple:
        """Create a consistent key for a comparison regardless of order"""
        return tuple(sorted([str(item_a), str(item_b)]))

    def get_next_comparison(self) -> Optional[Tuple[Union[int, str], Union[int, str]]]:
        """Get the next pair of items to compare without requiring immediate input.
        Returns None if no more comparisons are needed based on confidence threshold."""

        if hasattr(self, "min_confidence"):
            if not self.should_continue(self.min_confidence):
                return None

        # Calculate total possible unique comparisons
        total_possible = (len(self.items) * (len(self.items) - 1)) // 2

        # If we've compared all possible pairs, we're done
        if len(self.completed_comparisons) >= total_possible:
            return None

        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            # Use the same smart selection logic as generate_comparison_data
            progress = self.iteration_count / (
                len(self.items) * np.log(len(self.items))
            )
            if progress < 0.3:
                item_a, item_b = self.get_most_uncertain_pair()
            elif progress < 0.7:
                item_a, item_b = self.get_most_informative_pair()
            else:
                item_a, item_b = random.sample(self.items, 2)
                while item_a == item_b:
                    item_a, item_b = random.sample(self.items, 2)

            comparison_key = self.get_comparison_key(item_a, item_b)
            if comparison_key not in self.completed_comparisons:
                return item_a, item_b
            attempts += 1

        # If we're having trouble finding a new pair, try systematically
        for i, item_a in enumerate(self.items):
            for item_b in self.items[i + 1 :]:
                comparison_key = self.get_comparison_key(item_a, item_b)
                if comparison_key not in self.completed_comparisons:
                    return item_a, item_b

        # If we get here, something is wrong - we should have found an unused pair
        # or returned None earlier if all pairs were used
        return None

    def submit_comparison(
        self, item_a: Union[int, str], item_b: Union[int, str], response: int
    ) -> None:
        """Submit a comparison result for a pair of items.
        response should be:
        1 - if item_a is better than item_b
        2 - if they are equal
        3 - if item_b is better than item_a"""
        if response not in [1, 2, 3]:
            raise ValueError("Response must be 1, 2, or 3")

        self.update_single_query(item_a, item_b, response)
        self.completed_comparisons.add(self.get_comparison_key(item_a, item_b))
        self.history.clear()

    def generate_comparison_data(
        self, queries: int
    ) -> List[Tuple[Union[int, str], Union[int, str], float, float]]:
        comparison_data = []
        print(
            "Comparison commands: 1=yes, 2=tied, 3=second is better, p=print estimates, "
            "s=skip question, u=undo last comparison, q=quit"
        )

        i = 0
        while i < queries:
            # Check if we've reached sufficient confidence
            if hasattr(self, "min_confidence"):
                if not self.should_continue(self.min_confidence):
                    print("\nReached confidence threshold - stopping early!")
                    break

            if self.history:
                last_comparison = self.history[-1]
                item_a, item_b = last_comparison[0], last_comparison[1]
            else:
                attempts = 0
                max_attempts = 10
                while attempts < max_attempts:
                    if i < queries * 0.3:
                        item_a, item_b = self.get_most_uncertain_pair()
                    elif i < queries * 0.7:
                        item_a, item_b = self.get_most_informative_pair()
                    else:
                        item_a, item_b = random.sample(self.items, 2)
                        while item_a == item_b:
                            item_a, item_b = random.sample(self.items, 2)

                    comparison_key = self.get_comparison_key(item_a, item_b)
                    if comparison_key not in self.completed_comparisons:
                        break
                    attempts += 1

                if attempts == max_attempts:
                    self.completed_comparisons.clear()

            print(f"\nComparison {i+1}/{queries}")  # Show progress
            response = self.ask_question(item_a, item_b)
            if response == "skip":
                continue

            win_a, win_b = (
                (1, 0) if response == 1 else (0, 1) if response == 3 else (0.5, 0.5)
            )
            comparison_data.append((item_a, item_b, win_a, win_b))
            self.completed_comparisons.add(self.get_comparison_key(item_a, item_b))
            self.history.clear()
            i += 1

            # Show current rankings after each comparison if progress is enabled
            if hasattr(self, "show_progress") and self.show_progress:
                self.print_estimates()
        return comparison_data

    def update_ranks(
        self,
        comparison_data: List[Tuple[Union[int, str], Union[int, str], float, float]],
    ) -> None:
        """This method is now deprecated as updates happen in real-time"""
        pass

    def compute_ranks(self) -> Dict[Union[int, str], float]:
        return {
            player: alpha / (alpha + beta)
            for player, (alpha, beta) in self.alpha_beta.items()
        }

    def get_mean_uncertainty(self) -> float:
        """Calculate mean uncertainty across all items."""
        uncertainties = [
            self.standard_error(alpha, beta) for alpha, beta in self.alpha_beta.values()
        ]
        return sum(uncertainties) / len(uncertainties)

    def get_ranking_confidence(self) -> Dict[Union[int, str], float]:
        """Calculate confidence in ranking for each item (0-1 scale)."""
        total_comparisons = {item: 0.0 for item in self.items}
        for item, (alpha, beta) in self.alpha_beta.items():
            total_comparisons[item] = alpha + beta - 2  # Subtract initial values

        max_comparisons = max(total_comparisons.values())
        if max_comparisons == 0:
            return {item: 0.0 for item in self.items}

        # Combine number of comparisons and uncertainty into confidence score
        confidences = {}
        for item in self.items:
            uncertainty = self.standard_error(*self.alpha_beta[item])
            comparison_ratio = total_comparisons[item] / max_comparisons
            confidences[item] = (1 - uncertainty) * comparison_ratio

        return confidences

    def get_most_uncertain_pair(self) -> Tuple[Union[int, str], Union[int, str]]:
        """Get the pair of items with highest combined uncertainty that hasn't been compared."""
        uncertainties = {
            item: self.standard_error(alpha, beta)
            for item, (alpha, beta) in self.alpha_beta.items()
        }
        # Sort items by uncertainty
        sorted_items = sorted(uncertainties.items(), key=lambda x: x[1], reverse=True)

        # Find the first valid pair that hasn't been compared
        for i, (item_a, _) in enumerate(sorted_items):
            for item_b, _ in sorted_items[i + 1 :]:
                if (
                    self.get_comparison_key(item_a, item_b)
                    not in self.completed_comparisons
                ):
                    return item_a, item_b

        # If all pairs have been compared, return the first two items
        return sorted_items[0][0], sorted_items[1][0]

    def initialize_with_similarity(
        self, items: List[str], scores: Optional[Dict[str, float]] = None
    ) -> None:
        """Initialize rankings using text similarity when scores are available for some items"""
        if not scores:
            return

        # For items without scores, estimate based on similar items that have scores
        for item in items:
            if item not in scores:
                similar_items = [
                    (other_item, SequenceMatcher(None, item, other_item).ratio())
                    for other_item, score in scores.items()
                ]
                if similar_items:
                    most_similar = max(similar_items, key=lambda x: x[1])
                    if most_similar[1] > 0.8:  # Only use if similarity is high
                        scores[item] = scores[most_similar[0]]

    def should_continue(self, min_confidence: float = 0.9) -> bool:
        """Check if we should continue comparing based on confidence levels"""
        confidences = self.get_ranking_confidence()
        mean_confidence = sum(confidences.values()) / len(confidences)
        return mean_confidence < min_confidence  # Continue if confidence is too low

    def save_state(self, filename: str) -> None:
        """Save current state to file"""
        state = {
            "items": self.items,
            "alpha_beta": {str(k): v for k, v in self.alpha_beta.items()},
            "history": [
                (str(a), str(b), {str(k): v for k, v in s.items()})
                for a, b, s in self.history
            ],
        }
        with open(filename, "w") as f:
            json.dump(state, f)

    def load_state(self, filename: str) -> None:
        """Load state from file"""
        with open(filename, "r") as f:
            state = json.load(f)
        self.items = state["items"]
        self.alpha_beta = {
            eval(k) if k.isdigit() else k: tuple(v)
            for k, v in state["alpha_beta"].items()
        }
        self.history = [
            (
                eval(a) if a.isdigit() else a,
                eval(b) if b.isdigit() else b,
                {eval(k) if k.isdigit() else k: tuple(v) for k, v in s.items()},
            )
            for a, b, s in state["history"]
        ]

    def visualize_rankings(self) -> None:
        """Display a simple ASCII visualization of rankings"""
        ranks = self.compute_ranks()
        sorted_items = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        max_name_len = max(len(str(item)) for item in self.items)

        print("\nRanking visualization:")
        for item, rank in sorted_items:
            bar_length = int(rank * 40)
            print(
                f"{str(item):<{max_name_len}} | {'#' * bar_length}{' ' * (40-bar_length)} | {rank:.2f}"
            )

    def export_rankings(self, format: str = "csv") -> Union[str, Dict, pd.DataFrame]:
        """Export rankings in various formats"""
        ranks = self.compute_ranks()
        confidences = self.get_ranking_confidence()
        sorted_items = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

        if format == "json":
            return {
                "rankings": dict(sorted_items),
                "confidences": confidences,
                "metadata": {
                    "total_comparisons": self.iteration_count,
                    "mean_uncertainty": self.get_mean_uncertainty(),
                },
            }
        elif format == "markdown":
            lines = ["| Item | Rank | Confidence |", "|------|------|------------|"]
            for item, rank in sorted_items:
                lines.append(f"| {item} | {rank:.2f} | {confidences[item]:.2%} |")
            return "\n".join(lines)
        elif format == "csv":
            return pd.DataFrame(
                {
                    "Item": [item for item, _ in sorted_items],
                    "Rank": [rank for _, rank in sorted_items],
                    "Confidence": [confidences[item] for item, _ in sorted_items],
                }
            )
        else:
            raise ValueError(f"Unknown format: {format}")
