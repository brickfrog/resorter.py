import click
import random
import sys
from math import ceil, sqrt
from typing import Any, Dict, List, Optional, Tuple, Union
import os

import numpy as np
import pandas as pd


def read_input(data_input: str) -> pd.DataFrame:
    """
    Reads .csv into a dataframe, or splits a comma-separated string into a dataframe.
    """
    if os.path.exists(data_input) and data_input.endswith(".csv"):
        return pd.read_csv(data_input, header=None, dtype=str, na_filter=False)

    items = data_input.split(",")
    return pd.DataFrame(items, columns=["Items"])


def parse_input(
    df: pd.DataFrame,
) -> Tuple[List[Union[int, str]], Optional[Dict[Union[int, str], float]]]:
    """
    Parse the input dataframe to separate items and scores.
    """
    if df.shape[1] == 2:
        items = df.iloc[:, 0].tolist()
        scores = df.set_index(0).iloc[:, 0].to_dict()
    else:
        items = df.iloc[:, 0].tolist()
        scores = None  # No scores provided
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


class Config:
    def __init__(self, input, output, queries, levels, quantiles, progress):
        self.input = input
        self.output = output
        self.queries = queries
        self.levels = levels
        self.quantiles = quantiles
        self.progress = progress


class BradleyTerryModel:
    def __init__(
        self,
        items: List[Union[int, str]],
        scores: Optional[Dict[Union[int, str], float]] = None,
    ) -> None:
        self.items: List[Union[int, str]] = items
        self.alpha_beta: Dict[Union[int, str], Tuple[float, float]]
        if scores:
            self.alpha_beta = {
                item: (float(score), 1) for item, score in scores.items()
            }
        else:
            self.alpha_beta = {item: (1, 1) for item in items}

    @staticmethod
    def standard_error(alpha: float, beta: float) -> float:
        return sqrt((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1)))

    def bayesian_update(
        self, alpha: float, beta: float, win: float, lose: float
    ) -> Tuple[float, float]:
        return alpha + win, beta + lose

    def ask_question(
        self, item_a: Union[int, str], item_b: Union[int, str]
    ) -> Union[int, str]:
        while True:
            try:
                response = input(
                    f"Is '{click.style(item_a, fg='green')}' better than '{click.style(item_b, fg='green')}'? "  # noqa: E501
                )
                if response in ["1", "2", "3"]:
                    self.update_single_query(item_a, item_b, int(response))
                    return int(response)
                elif response == "s":
                    print("Skipping...")
                    return "skip"
                elif response == "p":
                    self.print_estimates()
                elif response == "q":
                    print("Quitting...")
                    exit(0)
                else:
                    print("Invalid input. Please enter 1, 2, 3, s, p, or q.")
            except ValueError:
                print("Invalid input. Please enter 1, 2, 3, s, p, or q.")

    def update_single_query(
        self, item_a: Union[int, str], item_b: Union[int, str], response: int
    ) -> None:
        winners = [(1, 0), (0.5, 0.5), (0, 1)]
        win_a, win_b = winners[response - 1]

        for item, win, lose in [(item_a, win_a, win_b), (item_b, win_b, win_a)]:
            alpha, beta = self.alpha_beta[item]
            self.alpha_beta[item] = self.bayesian_update(alpha, beta, win, lose)

    def print_estimates(self) -> None:
        mean_uncertainty = self.get_mean_uncertainty()
        confidences = self.get_ranking_confidence()
        
        print(f"\nMean uncertainty: {mean_uncertainty:.4f}")
        print("\nItem rankings:")
        sorted_items = sorted(
            self.alpha_beta.items(),
            key=lambda x: x[1][0] / (x[1][0] + x[1][1]),
            reverse=True
        )
        for item, (alpha, beta) in sorted_items:
            rank = alpha / (alpha + beta)
            se = self.standard_error(alpha, beta)
            confidence = confidences[item]
            print(f"{item}: rank = {rank:.2f}, σ = {se:.4f}, confidence = {confidence:.2%}")

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
                    
                # Consider both uncertainty and how close the items are in ranking
                rank_a = self.alpha_beta[item_a][0] / sum(self.alpha_beta[item_a])
                rank_b = self.alpha_beta[item_b][0] / sum(self.alpha_beta[item_b])
                rank_diff = abs(rank_a - rank_b)
                
                # Items with similar ranks but high uncertainty are most informative
                information_value = (uncertainties[item_a] + uncertainties[item_b]) * (1 - rank_diff)
                pairs.append((item_a, item_b, information_value))
        
        # Return the most informative pair
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[0][0], pairs[0][1]

    def generate_comparison_data(
        self, queries: int
    ) -> List[Tuple[Union[int, str], Union[int, str], float, float]]:
        comparison_data = []
        print(
            "Comparison commands: 1=yes, 2=tied, 3=second is better, p=print estimates, s=skip question, q=quit"
        )
        
        for i in range(queries):
            # Use different selection strategies based on progress
            if i < queries * 0.3:  # First 30%: focus on high uncertainty
                item_a, item_b = self.get_most_uncertain_pair()
            elif i < queries * 0.7:  # Middle 40%: use information gain
                item_a, item_b = self.get_most_informative_pair()
            else:  # Last 30%: random selection to avoid local optima
                item_a, item_b = random.sample(self.items, 2)
                while item_a == item_b:
                    item_a, item_b = random.sample(self.items, 2)
            
            response = self.ask_question(item_a, item_b)
            if response == "skip":
                continue
            
            win_a, win_b = (
                (1, 0) if response == 1 
                else (0, 1) if response == 3 
                else (0.5, 0.5)
            )
            comparison_data.append((item_a, item_b, win_a, win_b))
        return comparison_data

    def update_ranks(
        self,
        comparison_data: List[Tuple[Union[int, str], Union[int, str], float, float]],
    ) -> None:
        for item_a, item_b, win_a, win_b in comparison_data:
            alpha_a, beta_a = self.alpha_beta[item_a]
            alpha_b, beta_b = self.alpha_beta[item_b]
            self.alpha_beta[item_a] = self.bayesian_update(
                alpha_a, beta_a, win_a, win_b
            )
            self.alpha_beta[item_b] = self.bayesian_update(
                alpha_b, beta_b, win_b, win_a
            )

    def compute_ranks(self) -> Dict[Union[int, str], float]:
        return {
            player: alpha / (alpha + beta)
            for player, (alpha, beta) in self.alpha_beta.items()
        }

    def get_mean_uncertainty(self) -> float:
        """Calculate mean uncertainty across all items."""
        uncertainties = [
            self.standard_error(alpha, beta)
            for alpha, beta in self.alpha_beta.values()
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
        """Get the pair of items with highest combined uncertainty."""
        uncertainties = {
            item: self.standard_error(alpha, beta)
            for item, (alpha, beta) in self.alpha_beta.items()
        }
        # Sort items by uncertainty
        sorted_items = sorted(uncertainties.items(), key=lambda x: x[1], reverse=True)
        # Return the two most uncertain items
        return sorted_items[0][0], sorted_items[1][0]


@click.command()
@click.option(
    "--input",
    required=True,
    help="input file: a CSV file of items to sort: one per line, with up to two columns. (eg. both 'Akira' and 'Akira, 10' are valid)",  # noqa: E501
)
@click.option(
    "--output",
    required=False,
    help="output file: a file to write the final results to. Default: printing to stdout.",  # noqa: E501
)
@click.option(
    "--queries",
    default=None,
    type=int,
    help="Maximum number of questions to ask the user; defaults to N*log(N) comparisons. If already rated, 𝒪 (n) is a good max, but the more items and more levels in the scale and more accuracy desired, the more comparisons are needed.",  # noqa: E501
)
@click.option(
    "--levels",
    default=None,
    type=int,
    help="The highest level; rated items will be discretized into 1–l levels, so l=5 means items are bucketed into 5 levels: [1,2,3,4,5], etc. Maps onto quantiles; valid values: 2–100.",  # noqa: E501
)
@click.option(
    "--quantiles",
    default=None,
    type=str,
    help="What fraction to allocate to each level; space-separated; overrides `--levels`. This allows making one level of ratings narrower (and more precise) than the others, at their expense; for example, one could make 3-star ratings rarer with quantiles like `--quantiles '0 0.25 0.8 1'`. Default: uniform distribution (1--5 → '0.0 0.2 0.4 0.6 0.8 1.0').",  # noqa: E501
)
@click.option(
    "--progress",
    is_flag=True,
    help="Print the mean standard error to stdout",  # noqa: E501
)
def main(
    input: str,
    output: str,
    queries: Optional[int],
    levels: Optional[int],
    quantiles: Optional[str],
    progress: bool,
) -> None:
    config: Config = Config(input, output, queries, levels, quantiles, progress)
    
    try:
        df: pd.DataFrame = read_input(input)
    except FileNotFoundError:
        print("Input file not found.")
        return

    items, scores = parse_input(df)
    queries: int = determine_queries(items, config.queries)
    print(f"Number of queries: {queries}")

    model: BradleyTerryModel = BradleyTerryModel(items, scores)
    
    # Print initial state if progress tracking is enabled
    if config.progress:
        print("\nInitial state:")
        model.print_estimates()
    
    comparison_data = []
    for i in range(queries):
        if config.progress:
            print(f"\nQuery {i+1}/{queries}")
            if i > 0 and i % 5 == 0:  # Show progress every 5 comparisons
                model.print_estimates()
        
        new_comparison = model.generate_comparison_data(1)
        if new_comparison:  # Only update if we got a valid comparison
            comparison_data.extend(new_comparison)
            model.update_ranks(new_comparison)

    if config.progress:
        print("\nFinal rankings:")
        model.print_estimates()

    ranks: Dict[Any, float] = model.compute_ranks()
    sorted_ranks = {
        k: v for k, v in sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    }

    if config.levels:
        levels = assign_levels(sorted_ranks, config.levels)
        output_data = pd.DataFrame(
            {"Item": list(levels.keys()), "Quantiles": list(levels.values())}
        )
    elif config.quantiles:
        quantile_cutoffs = [float(x) for x in config.quantiles.split(" ")]
        quantiles = assign_custom_quantiles(sorted_ranks, quantile_cutoffs)
        output_data = pd.DataFrame(
            {"Item": list(quantiles.keys()), "Quantiles": list(quantiles.values())}
        )
    else:
        sorted_ranks = {k: v for k, v in reversed(list(sorted_ranks.items()))}
        level_assignments = {
            item: rank + 1 for rank, (item, _) in enumerate(sorted_ranks.items())
        }
        output_data = pd.DataFrame(
            list(level_assignments.items()), columns=["Item", "Quantiles"]
        )

    output_data = output_data.sort_values(by=["Quantiles"], ascending=False)

    if not config.output:
        output_data.to_csv(sys.stdout, index=False)
    else:
        output_data.to_csv(output, index=False)


if __name__ == "__main__":
    main()
