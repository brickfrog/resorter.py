from math import ceil
from typing import Dict, List, Optional, Tuple, Union, Sequence, Any, Mapping
import os
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
) -> Tuple[List[str], Optional[Dict[str, float]]]:
    """
    Parse the input dataframe to separate items and scores.
    Expects DataFrame with 'Item' and optionally 'Score' columns.
    """
    items = df["Item"].tolist()
    if "Score" in df.columns:
        scores = df.set_index("Item")["Score"].astype(float).to_dict()
    else:
        scores = None
    return items, scores


def determine_queries(items: Sequence[Union[int, str]], args_queries: Optional[int]) -> int:
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
    sorted_ranks: Dict[Any, float], quantile_cutoffs: List[float]
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
    sorted_ranks: Mapping[Any, float], num_levels: int
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


class BradleyTerryRanker:
    def __init__(
        self,
        items: Sequence[Union[int, str]],
        scores: Optional[Mapping[Any, Union[int, float]]] = None,
    ) -> None:
        self.items: Sequence[Union[int, str]] = items
        self.item_to_idx = {item: i for i, item in enumerate(items)}
        self.idx_to_item = {i: item for i, item in enumerate(items)}
        self.n_items = len(items)
        
        # Initialize parameters (log-strengths)
        if scores:
            # Use provided scores as initial values, ensuring they're positive
            raw_scores = np.array([float(scores.get(item, 1.0)) for item in items])
            # Ensure all scores are positive for log transformation
            raw_scores = np.maximum(raw_scores, 1e-6)  # Minimum positive value
            self.strengths = np.log(raw_scores / np.mean(raw_scores))
        else:
            # Initialize with small random values
            self.strengths = np.random.normal(0, 0.1, self.n_items)
            
        self.strengths -= np.mean(self.strengths)  # Normalize to zero mean
        
        self.comparison_matrix = np.zeros((self.n_items, self.n_items))
        self.win_matrix = np.zeros((self.n_items, self.n_items))
        self.iteration_count: int = 0
        self.completed_comparisons: set = set()
        self.history: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []

    def bradley_terry_prob(self, strength_i: float, strength_j: float) -> float:
        """Compute probability that item i beats item j"""
        return 1 / (1 + np.exp(-(strength_i - strength_j)))

    def log_likelihood(self, strengths: np.ndarray) -> float:
        """Compute log-likelihood of the model"""
        ll = 0
        eps = 1e-15  # Small constant to prevent log(0)
        for i in range(self.n_items):
            for j in range(self.n_items):
                if self.comparison_matrix[i, j] > 0:
                    p_ij = np.clip(self.bradley_terry_prob(strengths[i], strengths[j]), eps, 1-eps)
                    w_ij = self.win_matrix[i, j] / self.comparison_matrix[i, j]
                    ll += self.comparison_matrix[i, j] * (w_ij * np.log(p_ij) + (1 - w_ij) * np.log(1 - p_ij))
        return -ll  # Return negative for minimization

    def fit(self) -> None:
        """Fit the Bradley-Terry model using maximum likelihood"""
        if np.sum(self.comparison_matrix) == 0:
            return  # No comparisons to fit

        # Optimize with constraint that mean strength = 0
        constraints = {'type': 'eq', 'fun': lambda x: np.mean(x)}
        result = minimize(
            self.log_likelihood,
            self.strengths,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")
            
        self.strengths = result.x
        self.strengths -= np.mean(self.strengths)  # Ensure zero mean
        self.last_optimization_result = result  # Store for diagnostics

    def update_single_query(
        self, item_a: Union[int, str], item_b: Union[int, str], response: int
    ) -> None:
        """Update model with a single comparison result"""
        # Save state before update
        self.history.append((str(item_a), str(item_b), self.strengths.copy(), self.comparison_matrix.copy(), self.win_matrix.copy()))

        i, j = self.item_to_idx[item_a], self.item_to_idx[item_b]
        
        # Update comparison matrices
        self.comparison_matrix[i, j] += 1
        self.comparison_matrix[j, i] += 1
        
        # Convert response to win probability
        if response == 1:  # A wins
            win_prob = 1.0
        elif response == 2:  # Tie
            win_prob = 0.5
        else:  # B wins
            win_prob = 0.0
            
        self.win_matrix[i, j] += win_prob
        self.win_matrix[j, i] += (1 - win_prob)
        
        # Add to completed comparisons
        self.completed_comparisons.add(self.get_comparison_key(item_a, item_b))
        
        # Refit the model
        self.fit()
        self.iteration_count += 1

        # Check for inconsistency
        prob_a_beats_b = self.bradley_terry_prob(self.strengths[i], self.strengths[j])
        if abs(prob_a_beats_b - win_prob) > 0.7:
            print("\nWarning: This comparison seems inconsistent with previous rankings!")
            print("You might want to undo (u) and reconsider.")

    def undo_last_comparison(self) -> None:
        """Undo the last comparison by restoring the previous state"""
        if not self.history:
            print("Nothing to undo!")
            return

        item_a, item_b, previous_strengths, previous_comparison_matrix, previous_win_matrix = self.history.pop()
        i, j = self.item_to_idx[item_a], self.item_to_idx[item_b]
        
        # Restore previous state
        self.strengths = previous_strengths
        self.comparison_matrix = previous_comparison_matrix
        self.win_matrix = previous_win_matrix
        
        print(f"Undid comparison between '{item_a}' and '{item_b}'")

    def get_comparison_key(
        self, item_a: Union[int, str], item_b: Union[int, str]
    ) -> tuple:
        """Create a consistent key for a comparison regardless of order"""
        return tuple(sorted([str(item_a), str(item_b)]))

    def compute_ranks(self) -> Dict[Union[int, str], float]:
        """Convert log-strengths to probabilities"""
        # Scale the strengths to reduce extreme probabilities
        scaled_strengths = self.strengths / max(1.0, np.max(np.abs(self.strengths)))
        exp_strengths = np.exp(scaled_strengths)
        sum_exp_strengths = np.sum(exp_strengths)
        probs = exp_strengths / sum_exp_strengths
        return {self.idx_to_item[i]: float(p) for i, p in enumerate(probs)}

    def compute_ordinal_rankings(self) -> Dict[Union[int, str], int]:
        """Convert strengths to ordinal rankings (1st, 2nd, 3rd, etc.)"""
        # Sort by strength (highest first)
        sorted_indices = np.argsort(-self.strengths)  # Negative for descending order
        rankings = {}
        for rank, idx in enumerate(sorted_indices, 1):
            rankings[self.idx_to_item[idx]] = rank
        return rankings

    def get_uncertainty(self, item: Union[int, str]) -> float:
        """Compute uncertainty for an item using Fisher information and comparison count"""
        i = self.item_to_idx[item]
        total_comparisons = np.sum(self.comparison_matrix[i])
        if total_comparisons == 0:
            return 1.0
        
        # Use inverse Fisher information as uncertainty
        info = 0
        for j in range(self.n_items):
            if i != j and self.comparison_matrix[i, j] > 0:
                p_ij = self.bradley_terry_prob(self.strengths[i], self.strengths[j])
                info += self.comparison_matrix[i, j] * p_ij * (1 - p_ij)
        
        # Scale uncertainty by number of comparisons
        base_uncertainty = 1 / (1 + info)
        comparison_factor = np.exp(-total_comparisons / 3)  # Decay factor based on comparisons
        return base_uncertainty * comparison_factor

    def get_mean_uncertainty(self) -> float:
        """Calculate mean uncertainty across all items"""
        uncertainties = [self.get_uncertainty(item) for item in self.items]
        return sum(uncertainties) / len(uncertainties)

    def get_ranking_confidence(self) -> Dict[Union[int, str], float]:
        """Calculate confidence in ranking for each item (0-1 scale)"""
        total_comparisons = {
            item: np.sum(self.comparison_matrix[self.item_to_idx[item]])
            for item in self.items
        }
        max_comparisons = max(total_comparisons.values()) if total_comparisons else 1

        confidences = {}
        for item in self.items:
            uncertainty = self.get_uncertainty(item)
            # Scale confidence by number of comparisons and relative position certainty
            comparison_ratio = min(1.0, total_comparisons[item] / max_comparisons) if max_comparisons > 0 else 0.0
            
            # Get relative position certainty
            rank_certainties = []
            for other_item in self.items:
                if other_item != item:
                    p_win = self.bradley_terry_prob(
                        self.strengths[self.item_to_idx[item]],
                        self.strengths[self.item_to_idx[other_item]]
                    )
                    rank_certainties.append(abs(p_win - 0.5) * 2)  # Scale to [0,1]
            position_certainty = np.mean(rank_certainties) if rank_certainties else 0.0
            
            # Combine all factors
            confidence = float((1 - uncertainty) * comparison_ratio * (0.5 + 0.5 * position_certainty))
            confidences[item] = max(0.0, min(1.0, confidence))  # Ensure bounds

        return confidences

    def get_most_informative_pair(self) -> Tuple[Optional[Union[int, str]], Optional[Union[int, str]]]:
        """Get the pair of items that would provide the most information"""
        # Get all possible pairs and their information values
        pairs = []
        total_possible_pairs = (self.n_items * (self.n_items - 1)) // 2
        
        # If we've completed all possible pairs, clear and start over
        if len(self.completed_comparisons) >= total_possible_pairs:
            self.completed_comparisons.clear()

        for i, item_a in enumerate(self.items):
            for j, item_b in enumerate(self.items[i+1:], i+1):
                key = self.get_comparison_key(item_a, item_b)
                if key in self.completed_comparisons:
                    continue
                    
                # Compute expected information gain
                p_ij = self.bradley_terry_prob(self.strengths[i], self.strengths[j])
                uncertainty_i = self.get_uncertainty(item_a)
                uncertainty_j = self.get_uncertainty(item_b)
                
                # Compute information value based on:
                # 1. Combined uncertainty of items
                # 2. How close to 50/50 the predicted outcome is
                # 3. Number of comparisons each item has
                uncertainty_factor = (uncertainty_i + uncertainty_j)
                prediction_factor = 1 - abs(p_ij - 0.5)
                comparison_counts = (
                    np.sum(self.comparison_matrix[i]) +
                    np.sum(self.comparison_matrix[j])
                )
                comparison_factor = np.exp(-comparison_counts / 6)  # Prefer less compared pairs
                
                info_value = uncertainty_factor * prediction_factor * comparison_factor
                info_value += np.random.normal(0, 0.001)  # Tiny random noise to break ties
                pairs.append((item_a, item_b, info_value))

        if not pairs:
            # If still no pairs after clearing, we're done
            return None, None

        # Sort by information value and return the most informative pair
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[0][0], pairs[0][1]

    def get_most_uncertain_pair(self) -> Tuple[Optional[Union[int, str]], Optional[Union[int, str]]]:
        """Get the pair of items with highest combined uncertainty"""
        uncertainties = [(item, self.get_uncertainty(item)) for item in self.items]
        # Add small random noise to break ties
        uncertainties = [(item, unc + np.random.normal(0, 0.01)) for item, unc in uncertainties]
        sorted_items = sorted(uncertainties, key=lambda x: x[1], reverse=True)

        for i, (item_a, _) in enumerate(sorted_items):
            for item_b, _ in sorted_items[i+1:]:
                if self.get_comparison_key(item_a, item_b) not in self.completed_comparisons:
                    return item_a, item_b

        # If all pairs are completed, return None
        return None, None

    def should_continue(self, min_confidence: float = 0.9) -> bool:
        """Check if we should continue comparing based on confidence levels"""
        if self.iteration_count == 0:
            return True
            
        # Get confidences for all items
        confidences = self.get_ranking_confidence()
        mean_confidence = sum(confidences.values()) / len(confidences)
        
        # Check if we have enough comparisons and confidence
        min_comparisons_per_item = 3  # Require at least 3 comparisons per item
        comparison_counts = [
            np.sum(self.comparison_matrix[self.item_to_idx[item]])
            for item in self.items
        ]
        
        # Continue if either:
        # 1. Not all items have minimum comparisons
        if any(count < min_comparisons_per_item for count in comparison_counts):
            return True
            
        # 2. Mean confidence is below threshold and we haven't compared everything multiple times
        if mean_confidence < min_confidence:
            # Only continue if we haven't done too many comparisons
            max_comparisons = self.n_items * (self.n_items - 1)  # Maximum possible unique comparisons
            total_comparisons = sum(comparison_counts) // 2  # Divide by 2 as each comparison is counted twice
            return total_comparisons < max_comparisons * 2  # Allow each pair to be compared twice
            
        return False

    def save_state(self, filename: str) -> None:
        """Save current state to file"""
        state = {
            "items": self.items,
            "strengths": self.strengths.tolist(),
            "comparison_matrix": self.comparison_matrix.tolist(),
            "win_matrix": self.win_matrix.tolist(),
            "history": [(str(a), str(b), s.tolist(), cm.tolist(), wm.tolist()) for a, b, s, cm, wm in self.history],
            "iteration_count": self.iteration_count,
            "completed_comparisons": [list(comp) for comp in self.completed_comparisons],
        }
        with open(filename, "w") as f:
            json.dump(state, f)

    def load_state(self, filename: str) -> None:
        """Load state from file"""
        with open(filename, "r") as f:
            state = json.load(f)
        self.items = state["items"]
        self.item_to_idx = {item: i for i, item in enumerate(self.items)}
        self.idx_to_item = {i: item for i, item in enumerate(self.items)}
        self.n_items = len(self.items)
        self.strengths = np.array(state["strengths"])
        self.comparison_matrix = np.array(state["comparison_matrix"])
        self.win_matrix = np.array(state["win_matrix"])
        self.history = [
            (
                eval(a) if a.isdigit() else a,
                eval(b) if b.isdigit() else b,
                np.array(s),
                np.array(cm),
                np.array(wm),
            )
            for a, b, s, cm, wm in state["history"]
        ]
        self.iteration_count = state.get("iteration_count", 0)
        # Convert lists back to tuples for completed_comparisons
        self.completed_comparisons = set(tuple(comp) for comp in state.get("completed_comparisons", []))

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

    def model_diagnostics(self) -> Dict[str, float]:
        """Compute model diagnostics similar to R's BradleyTerry2"""
        if np.sum(self.comparison_matrix) == 0:
            return {"log_likelihood": 0.0, "aic": 0.0, "deviance": 0.0}
            
        # Log-likelihood
        ll = -self.log_likelihood(self.strengths)
        
        # AIC (Akaike Information Criterion)
        k = self.n_items - 1  # Number of free parameters (one constraint)
        aic = 2 * k - 2 * ll
        
        # Deviance (saturated model comparison)
        saturated_ll = 0
        for i in range(self.n_items):
            for j in range(self.n_items):
                if self.comparison_matrix[i, j] > 0:
                    w_ij = self.win_matrix[i, j] / self.comparison_matrix[i, j]
                    if w_ij > 0 and w_ij < 1:
                        saturated_ll += self.comparison_matrix[i, j] * (
                            w_ij * np.log(w_ij) + (1 - w_ij) * np.log(1 - w_ij)
                        )
        
        deviance = 2 * (saturated_ll - ll)
        
        return {
            "log_likelihood": ll,
            "aic": aic,
            "deviance": deviance,
            "n_comparisons": int(np.sum(self.comparison_matrix) / 2),
            "mean_strength": float(np.mean(self.strengths)),
            "strength_variance": float(np.var(self.strengths))
        }
