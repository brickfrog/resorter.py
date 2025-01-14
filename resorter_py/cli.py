import click
import sys
import json
import pandas as pd
from typing import Any, Dict, Optional

from .ranker import (
    BayesianPairwiseRanker,
    read_input,
    parse_input,
    determine_queries,
    assign_levels,
    assign_custom_quantiles,
)


class Config:
    def __init__(
        self,
        input,
        output,
        queries,
        levels,
        quantiles,
        progress,
        save_state,
        load_state,
        min_confidence,
        visualize,
        format="csv",
    ):
        self.input = input
        self.output = output
        self.queries = queries
        self.levels = levels
        self.quantiles = quantiles
        self.progress = progress
        self.save_state = save_state
        self.load_state = load_state
        self.min_confidence = min_confidence
        self.visualize = visualize
        self.format = format


@click.command()
@click.option(
    "--input",
    required=True,
    help="input file: a CSV file of items to sort: one per line, with up to two columns. (eg. both 'Akira' and 'Akira, 10' are valid)",
)
@click.option(
    "--output",
    required=False,
    help="output file: a file to write the final results to. Default: printing to stdout.",
)
@click.option(
    "--queries",
    default=None,
    type=int,
    help="Maximum number of questions to ask the user; defaults to N*log(N) comparisons.",
)
@click.option(
    "--levels",
    default=None,
    type=int,
    help="The highest level; rated items will be discretized into 1–l levels.",
)
@click.option(
    "--quantiles",
    default=None,
    type=str,
    help="What fraction to allocate to each level; space-separated; overrides `--levels`.",
)
@click.option(
    "--progress",
    is_flag=True,
    help="Print the mean standard error to stdout",
)
@click.option(
    "--save-state",
    type=str,
    help="Save the current state to this file",
)
@click.option(
    "--load-state",
    type=str,
    help="Load the previous state from this file",
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.9,
    help="Minimum confidence level before stopping (0-1)",
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Show ASCII visualization of rankings",
)
@click.option(
    "--format",
    type=click.Choice(["csv", "json", "markdown"]),
    default="csv",
    help="Output format for the rankings",
)
def main(
    input: str,
    output: str,
    queries: Optional[int],
    levels: Optional[int],
    quantiles: Optional[str],
    progress: bool,
    save_state: Optional[str],
    load_state: Optional[str],
    min_confidence: float,
    visualize: bool,
    format: str,
) -> None:
    config: Config = Config(
        input,
        output,
        queries,
        levels,
        quantiles,
        progress,
        save_state,
        load_state,
        min_confidence,
        visualize,
        format,
    )

    try:
        df: pd.DataFrame = read_input(input)
    except FileNotFoundError:
        print("Input file not found.")
        return

    items, scores = parse_input(df)
    queries: int = determine_queries(items, config.queries)
    print(f"Number of queries: {queries}")

    model: BayesianPairwiseRanker = BayesianPairwiseRanker(items, scores)
    model.show_progress = config.progress
    model.min_confidence = config.min_confidence

    if config.progress:
        print("\nInitial state:")
        model.print_estimates()

    # Run the comparison process
    model.generate_comparison_data(queries)

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
            {
                "Item": list(level_assignments.keys()),
                "Quantiles": list(level_assignments.values()),
            }
        )

    output_data = output_data.sort_values(by=["Quantiles"], ascending=False)

    # Export in the requested format
    output_data = model.export_rankings(format)

    if not config.output:
        if format == "json":
            print(json.dumps(output_data, indent=2))
        elif format == "markdown":
            print(output_data)
        else:
            output_data.to_csv(sys.stdout, index=False)
    else:
        if format == "json":
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)
        elif format == "markdown":
            with open(output, "w") as f:
                f.write(output_data)
        else:
            output_data.to_csv(output, index=False)

    # Add visualization if requested
    if config.visualize:
        model.visualize_rankings()

    # Save state if requested
    if config.save_state:
        model.save_state(config.save_state)


if __name__ == "__main__":
    main()
