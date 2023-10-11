import argparse
import random
import signal
import sys
from math import ceil, sqrt
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def parse_input(df: pd.DataFrame) -> Tuple[List[Union[int, str]], Optional[Dict[Union[int, str], float]]]:
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
    return int(ceil(list_length * np.log(list_length) + 1))

def generate_bin_edges(data: List[float], levels: Optional[int] = None, quantiles: Optional[int] = None) -> Optional[np.ndarray]:
    if quantiles is not None:
        return np.quantile(data, np.linspace(0, 1, quantiles + 1))
    elif levels is not None:
        return np.linspace(min(data), max(data), levels + 1)
    else:
        return None

def assign_custom_quantiles(sorted_ranks: Dict[Union[int, str], float], quantile_cutoffs: List[float]) -> Dict[Union[int, str], int]:
    sorted_values = [val for _, val in sorted_ranks.items()]
    num_items = len(sorted_values)
    cutoff_positions = [int(c * num_items) for c in quantile_cutoffs]
    quantiles = {}
    quantile_label = 1
    for i, (key, _) in enumerate(sorted_ranks.items()):
        if i >= cutoff_positions[quantile_label]:
            quantile_label += 1
        quantiles[key] = len(quantile_cutoffs) - quantile_label
    return quantiles

def assign_levels(sorted_ranks: Dict[Union[int, str], float], num_levels: int) -> Dict[Union[int, str], int]:
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
    def __init__(self, args):
        self.input = args.input
        self.output = args.output
        self.queries = args.queries
        self.levels = args.levels
        self.quantiles = args.quantiles
        self.progress = args.progress


class BradleyTerryModel:
    def __init__(self, items: List[Union[int, str]], scores: Optional[Dict[Union[int, str], float]] = None) -> None:
        self.items: List[Union[int, str]] = items
        self.alpha_beta: Dict[Union[int, str], Tuple[float, float]]
        if scores:
            self.alpha_beta = {item: (score, 1) for item, score in scores.items()}
        else:
            self.alpha_beta = {item: (1, 1) for item in items}
        
        self.in_sigtstp_state: bool = False
        signal.signal(signal.SIGINT, self.sigint_handler)
        signal.signal(signal.SIGTSTP, self.sigtstp_handler)

    @staticmethod
    def standard_error(alpha: float, beta: float) -> float:
        return sqrt((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1)))

    def sigtstp_handler(self, signum: int, frame: Any) -> None:
        if self.in_sigtstp_state:
            print("\nCtrl+Z pressed again. Exiting.")
            self.in_sigtstp_state = False
            sys.exit(0)
        else:
            print(
                "\nCtrl+Z pressed. You can either exit with q/Ctrl+Z with Ctrl+C or continue answering."
            )
            self.in_sigtstp_state = True

    def sigint_handler(self, signum: int, frame: Any) -> None:
        if self.in_sigtstp_state:
            print("\nContinuing to answer questions...")
            return
        else:
            print("\nCtrl+C pressed. Cleaning up before exit.")
            sys.exit(0) 

    def bayesian_update(self, alpha: float, beta: float, win: float, lose: float) -> Tuple[float, float]:
        return alpha + win, beta + lose

    def ask_question(self, item_a: Union[int, str], item_b: Union[int, str]) -> int:
        while True:
            try:
                print(f"Compare {item_a} to {item_b}")
                response = input(
                    "1 for A better, 2 for B better, 3 for tie, p for print estimates, q to quit: "
                )
                if response in ["1", "2", "3"]:
                    self.update_single_query(item_a, item_b, int(response))
                    return int(response)
                elif response == "p":
                    self.print_estimates()
                elif response == "q":
                    print("Quitting...")
                    exit(0)
                else:
                    print("Invalid input. Please enter 1, 2, 3, or p.")
            except ValueError:
                print("Invalid input. Please enter 1, 2, 3, or p.")

    def update_single_query(self, item_a: Union[int, str], item_b: Union[int, str], response: int) -> None:
        winners = [(1, 0), (0, 1), (0.5, 0.5)]
        win_a, win_b = winners[response - 1]

        for item, win, lose in [(item_a, win_a, win_b), (item_b, win_b, win_a)]:
            alpha, beta = self.alpha_beta[item]
            self.alpha_beta[item] = self.bayesian_update(alpha, beta, win, lose)

    def print_estimates(self) -> None:
        for item, (alpha, beta) in self.alpha_beta.items():
            rank = alpha / (alpha + beta)
            se = standard_error(alpha, beta)
            print(f"{item}: rank = {round(rank,2)}, σ = {round(se,4)}")

    def generate_comparison_data(self, queries: int) -> List[Tuple[Union[int, str], Union[int, str], float, float]]:
        comparison_data = []
        for _ in range(queries):
            item_a, item_b = random.sample(self.items, 2)
            response = self.ask_question(item_a, item_b)
            win_a, win_b = (
                (1, 0) if response == 1 else (0, 1) if response == 2 else (0.5, 0.5)
            )
            comparison_data.append((item_a, item_b, win_a, win_b))
        return comparison_data

    def update_ranks(self, comparison_data: List[Tuple[Union[int, str], Union[int, str], float, float]]) -> None:
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


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="input file: a CSV file of items to sort: one per line, with up to two columns. (eg. both 'Akira' and 'Akira, 10' are valid)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output file: a file to write the final results to. Default: printing to stdout.",
    )
    parser.add_argument(
        "--queries",
        default=None,
        type=int,
        help="Maximum number of questions to ask the user; defaults to N*log(N) comparisons. If already rated, 𝒪 (n) is a good max, but the more items and more levels in the scale and more accuracy desired, the more comparisons are needed.",
    )
    parser.add_argument(
        "--levels",
        default=None,
        type=int,
        help="The highest level; rated items will be discretized into 1–l levels, so l=5 means items are bucketed into 5 levels: [1,2,3,4,5], etc. Maps onto quantiles; valid values: 2–100.",
    )
    parser.add_argument(
        "--quantiles",
        default=None,
        type=str,
        help="What fraction to allocate to each level; space-separated; overrides `--levels`. This allows making one level of ratings narrower (and more precise) than the others, at their expense; for example, one could make 3-star ratings rarer with quantiles like `--quantiles '0 0.25 0.8 1'`. Default: uniform distribution (1--5 → '0.0 0.2 0.4 0.6 0.8 1.0').",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print the mean standard error to stdout",
    )
    # TODO: no-scale / verbose

    args: argparse.Namespace = parser.parse_args()
    config: Config = Config(args)

    try:
        df: pd.DataFrame = pd.read_csv(args.input, header=None)
    except FileNotFoundError:
        print("Input file not found.")
        return

    items, scores = parse_input(df)
    queries: int = determine_queries(items, config.queries)
    print(f"Number of queries: {queries}")

    model: BradleyTerryModel = BradleyTerryModel(items, scores)
    if args.progress:
        model.print_estimates()

    comparison_data = model.generate_comparison_data(queries)
    model.update_ranks(comparison_data)

    if args.progress:
        model.print_estimates()

    ranks: Dict[Any, float] = model.compute_ranks()
    sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda x: x[1], reverse=True)}

    if args.levels:
        levels = assign_levels(sorted_ranks, args.levels)
        output_data = pd.DataFrame({"Item": list(levels.keys()), "Quantiles": list(levels.values())})
    elif args.quantiles:
        quantile_cutoffs = [float(x) for x in args.quantiles.split(" ")]
        quantiles = assign_custom_quantiles(sorted_ranks, quantile_cutoffs)
        output_data = pd.DataFrame({"Item": list(quantiles.keys()), "Quantiles": list(quantiles.values())})
    else:
        sorted_ranks = {k: v for k, v in reversed(list(sorted_ranks.items()))}
        level_assignments = {item: rank + 1 for rank, (item, _) in enumerate(sorted_ranks.items())}
        output_data = pd.DataFrame(list(level_assignments.items()), columns=["Item", "Quantiles"])

    output_data = output_data.sort_values(by=["Quantiles"], ascending=False)
    output_data.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
