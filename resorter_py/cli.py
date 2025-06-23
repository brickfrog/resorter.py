import click
import sys
import json
import pandas as pd
from typing import Optional

from .ranker import (
    BradleyTerryRanker,
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
        confidence_intervals,
        diagnostics,
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
        self.confidence_intervals = confidence_intervals
        self.diagnostics = diagnostics
        self.visualize = visualize
        self.format = format


@click.command()
@click.option(
    "--input",
    "input_file",
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
    help="Print the mean uncertainty and model diagnostics during ranking",
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
    "--confidence-intervals",
    is_flag=True,
    help="Include confidence intervals in output",
)
@click.option(
    "--diagnostics",
    is_flag=True,
    help="Show model diagnostics (AIC, log-likelihood, etc.)",
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
    input_file: str,
    output: str,
    queries: Optional[int],
    levels: Optional[int],
    quantiles: Optional[str],
    progress: bool,
    save_state: Optional[str],
    load_state: Optional[str],
    min_confidence: float,
    confidence_intervals: bool,
    diagnostics: bool,
    visualize: bool,
    format: str,
) -> None:
    config: Config = Config(
        input_file,
        output,
        queries,
        levels,
        quantiles,
        progress,
        save_state,
        load_state,
        min_confidence,
        confidence_intervals,
        diagnostics,
        visualize,
        format,
    )

    try:
        df: pd.DataFrame = read_input(input_file)
    except FileNotFoundError:
        print("Input file not found.")
        return

    items, scores = parse_input(df)
    num_queries: int = determine_queries(items, config.queries)
    print(f"Number of queries: {num_queries}")

    model = BradleyTerryRanker(items, scores)
    if config.load_state:
        try:
            model.load_state(config.load_state)
            print(f"Loaded state from {config.load_state}")
        except FileNotFoundError:
            print(f"State file {config.load_state} not found, starting fresh.")
        except json.JSONDecodeError:
            print(f"Invalid state file {config.load_state}, starting fresh.")

    # Run the comparison process
    i = 0
    print(
        "Comparison commands: 1=yes, 2=tied, 3=second is better, p=print estimates, "
        "s=skip question, u=undo last comparison, q=quit"
    )

    while i < num_queries:
        if hasattr(model, "min_confidence"):
            if not model.should_continue(config.min_confidence):
                print("\nReached confidence threshold - stopping early!")
                break

        # Get next comparison
        item_a, item_b = model.get_most_informative_pair()
        if item_a is None or item_b is None:
            print("\nNo more pairs to compare!")
            break
        
        # Ask for comparison
        print(f"\nComparison {i+1}/{num_queries}")
        while True:
            try:
                response = input(f"Is '{item_a}' better than '{item_b}'? ")
                if response in ["1", "2", "3"]:
                    model.update_single_query(item_a, item_b, int(response))  # Ensure item_a and item_b are valid types
                    i += 1
                    break
                elif response == "s":
                    print("Skipping...")
                    break
                elif response == "p":
                    ranks = model.compute_ranks()
                    confidences = model.get_ranking_confidence()
                    print("\nCurrent rankings:")
                    for item, rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
                        confidence = confidences[item]
                        uncertainty = model.get_uncertainty(item)
                        print(f"{item}: rank = {rank:.6f}, confidence = {confidence:.2%}, uncertainty = {uncertainty:.3f}")
                    
                    if config.confidence_intervals:
                        print("\nConfidence intervals (95%):")
                        ci = model.get_confidence_intervals()
                        for item, (lower, upper) in ci.items():
                            strength = model.strengths[model.item_to_idx[item]]
                            print(f"{item}: {strength:.3f} [{lower:.3f}, {upper:.3f}]")
                    
                    if config.diagnostics:
                        print("\nModel diagnostics:")
                        diag = model.model_diagnostics()
                        print(f"  Log-likelihood: {diag['log_likelihood']:.3f}")
                        print(f"  AIC: {diag['aic']:.3f}")
                        print(f"  Deviance: {diag['deviance']:.3f}")
                        print(f"  Comparisons: {diag['n_comparisons']}")
                        print(f"  Mean strength: {diag['mean_strength']:.3f}")
                        print(f"  Strength variance: {diag['strength_variance']:.3f}")
                elif response == "u":
                    model.undo_last_comparison()
                    i = max(0, i - 1)
                    break
                elif response == "q":
                    print("Quitting...")
                    break
                else:
                    print("Invalid input. Please enter 1, 2, 3, s, p, u, or q.")
            except ValueError:
                print("Invalid input. Please enter 1, 2, 3, s, p, u, or q.")

        if response == "q":
            break

        if config.progress:
            ranks = model.compute_ranks()
            confidences = model.get_ranking_confidence()
            mean_uncertainty = model.get_mean_uncertainty()
            print(f"\nMean uncertainty: {mean_uncertainty:.4f}")
            
            if config.diagnostics and model.iteration_count > 0:
                diag = model.model_diagnostics()
                print(f"Model fit - AIC: {diag['aic']:.2f}, Log-likelihood: {diag['log_likelihood']:.3f}")
            
            print("\nCurrent rankings:")
            for item, rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
                confidence = confidences[item]
                uncertainty = model.get_uncertainty(item)
                print(f"{item}: rank = {rank:.6f}, confidence = {confidence:.2%}, uncertainty = {uncertainty:.3f}")

    # Get final rankings
    ranks = model.compute_ranks()
    confidences = model.get_ranking_confidence()
    sorted_ranks = {
        k: v for k, v in sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    }

    # Create base DataFrame with rankings
    output_data = pd.DataFrame({
        "Item": list(sorted_ranks.keys()),
        "Rank": list(sorted_ranks.values()),
        "Confidence": [confidences[item] for item in sorted_ranks.keys()]
    })

    # Add uncertainty information
    uncertainties = {item: model.get_uncertainty(item) for item in sorted_ranks.keys()}
    output_data["Uncertainty"] = [uncertainties[item] for item in output_data["Item"]]

    # Add confidence intervals if requested
    if config.confidence_intervals:
        ci = model.get_confidence_intervals()
        output_data["CI_Lower"] = [ci[item][0] for item in output_data["Item"]]
        output_data["CI_Upper"] = [ci[item][1] for item in output_data["Item"]]
        output_data["Strength"] = [model.strengths[model.item_to_idx[item]] for item in output_data["Item"]]

    # Add level/quantile information if requested
    if config.levels:
        level_assignments = assign_levels(sorted_ranks, config.levels)
        output_data["Level"] = [level_assignments[item] for item in output_data["Item"]]
    elif config.quantiles:
        quantile_cutoffs = [float(x) for x in config.quantiles.split(" ")]
        quantile_assignments = assign_custom_quantiles(sorted_ranks, quantile_cutoffs)
        output_data["Quantile"] = [quantile_assignments[item] for item in output_data["Item"]]

    # Format output based on requested format
    if format == "json":
        result = {
            "rankings": dict(zip(output_data["Item"], output_data["Rank"])),
            "confidences": dict(zip(output_data["Item"], output_data["Confidence"])),
            "uncertainties": dict(zip(output_data["Item"], output_data["Uncertainty"])),
            "metadata": {
                "total_comparisons": model.iteration_count,
                "mean_uncertainty": model.get_mean_uncertainty(),
            }
        }
        
        if config.confidence_intervals:
            result["confidence_intervals"] = {
                item: {"lower": row["CI_Lower"], "upper": row["CI_Upper"], "strength": row["Strength"]}
                for item, (_, row) in zip(output_data["Item"], output_data.iterrows())
            }
            
        if config.diagnostics:
            result["model_diagnostics"] = model.model_diagnostics()
            
        if config.levels:
            result["levels"] = dict(zip(output_data["Item"], output_data["Level"]))
        elif config.quantiles:
            result["quantiles"] = dict(zip(output_data["Item"], output_data["Quantile"]))
            
    elif format == "markdown":
        headers = ["Item", "Rank", "Confidence", "Uncertainty"]
        if config.confidence_intervals:
            headers.extend(["Strength", "CI_Lower", "CI_Upper"])
        if "Level" in output_data.columns:
            headers.append("Level")
        elif "Quantile" in output_data.columns:
            headers.append("Quantile")
        
        lines = [" | ".join(headers), "|".join(["-" * len(h) for h in headers])]
        for _, row in output_data.iterrows():
            values = [str(row["Item"]), f"{row['Rank']:.2f}", f"{row['Confidence']:.2%}", f"{row['Uncertainty']:.3f}"]
            if config.confidence_intervals:
                values.extend([f"{row['Strength']:.3f}", f"{row['CI_Lower']:.3f}", f"{row['CI_Upper']:.3f}"])
            if "Level" in output_data.columns:
                values.append(str(row["Level"]))
            elif "Quantile" in output_data.columns:
                values.append(str(row["Quantile"]))
            lines.append(" | ".join(values))
        result = "\n".join(lines)
    else:  # csv
        result = output_data

    # Export in the requested format
    if not config.output:
        if format == "json":
            print(json.dumps(result, indent=2))
        elif format == "markdown":
            print(result)
        else:
            output_data.to_csv(sys.stdout, index=False)
    else:
        if format == "json":
            with open(output, "w") as f:
                json.dump(result, f, indent=2)
        elif format == "markdown":
            with open(output, "w") as f:
                f.write(str(result))
        else:
            output_data.to_csv(output, index=False)

    # Add visualization if requested
    if config.visualize:
        model.visualize_rankings()

    # Show final diagnostics if requested
    if config.diagnostics:
        print("\n" + "="*50)
        print("FINAL MODEL DIAGNOSTICS")
        print("="*50)
        diag = model.model_diagnostics()
        ordinal_ranks = model.compute_ordinal_rankings()
        
        print("Model Performance:")
        print(f"  Log-likelihood: {diag['log_likelihood']:.3f}")
        print(f"  AIC: {diag['aic']:.3f}")
        print(f"  Deviance: {diag['deviance']:.3f}")
        print(f"  Total comparisons: {diag['n_comparisons']}")
        print(f"  Mean uncertainty: {model.get_mean_uncertainty():.4f}")
        
        print("\nParameter Statistics:")
        print(f"  Mean strength: {diag['mean_strength']:.6f}")
        print(f"  Strength variance: {diag['strength_variance']:.3f}")
        
        print("\nFinal Rankings (Ordinal):")
        for item, rank in sorted(ordinal_ranks.items(), key=lambda x: x[1]):
            uncertainty = model.get_uncertainty(item)
            print(f"  {rank}. {item} (uncertainty: {uncertainty:.3f})")

    # Save state if requested
    if config.save_state:
        model.save_state(config.save_state)
        print(f"Saved state to {config.save_state}")


if __name__ == "__main__":
    main()
