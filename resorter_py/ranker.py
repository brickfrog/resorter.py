from math import ceil
from typing import Dict, List, Optional, Tuple, Union, Sequence, Any, Mapping, TYPE_CHECKING
import csv
import os
import json

import numpy as np
from scipy.optimize import minimize

try:
    import pandas as pd
except ModuleNotFoundError:  # Optional dependency for DataFrame I/O.
    pd = None

if TYPE_CHECKING:
    import pandas as pd


class StateValidationError(ValueError):
    """Raised when a state file contains invalid or corrupted data."""


_REQUIRED_STATE_KEYS = {
    "items",
    "strengths",
    "comparison_matrix",
    "win_matrix",
    "iteration_count",
    "completed_comparisons",
}


def _validate_state(state: dict) -> None:
    """Validate a loaded state dict before assigning to the model.

    Raises StateValidationError on any structural or semantic problem.
    """
    if not isinstance(state, dict):
        raise StateValidationError("State must be a JSON object")

    # --- required keys ---
    missing = _REQUIRED_STATE_KEYS - state.keys()
    if missing:
        raise StateValidationError(f"Missing required keys: {sorted(missing)}")

    # --- items ---
    items = state["items"]
    if not isinstance(items, list) or len(items) == 0:
        raise StateValidationError("'items' must be a non-empty list")
    if not all(isinstance(i, str) for i in items):
        raise StateValidationError("'items' must contain only strings")
    n = len(items)

    # --- strengths ---
    strengths = state["strengths"]
    if not isinstance(strengths, list) or len(strengths) != n:
        raise StateValidationError(
            f"'strengths' must be a list of length {n}, got length {len(strengths) if isinstance(strengths, list) else type(strengths).__name__}"
        )
    if not all(isinstance(v, (int, float)) for v in strengths):
        raise StateValidationError("'strengths' must contain only numbers")

    # --- matrix helpers ---
    def _check_matrix(name: str, matrix: object) -> None:
        if not isinstance(matrix, list) or len(matrix) != n:
            raise StateValidationError(
                f"'{name}' must be a {n}x{n} list of lists"
            )
        for row_idx, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != n:
                raise StateValidationError(
                    f"'{name}' row {row_idx} must have length {n}"
                )
            for col_idx, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise StateValidationError(
                        f"'{name}[{row_idx}][{col_idx}]' must be a number"
                    )
                if val < 0:
                    raise StateValidationError(
                        f"'{name}[{row_idx}][{col_idx}]' must be >= 0, got {val}"
                    )

    _check_matrix("comparison_matrix", state["comparison_matrix"])
    _check_matrix("win_matrix", state["win_matrix"])

    # --- win_matrix <= comparison_matrix ---
    cm = state["comparison_matrix"]
    wm = state["win_matrix"]
    for i in range(n):
        for j in range(n):
            if wm[i][j] > cm[i][j]:
                raise StateValidationError(
                    f"win_matrix[{i}][{j}] ({wm[i][j]}) exceeds comparison_matrix[{i}][{j}] ({cm[i][j]})"
                )

    # --- iteration_count ---
    ic = state["iteration_count"]
    if not isinstance(ic, int) or ic < 0:
        raise StateValidationError(
            "'iteration_count' must be a non-negative integer"
        )

    # --- completed_comparisons ---
    cc = state["completed_comparisons"]
    if not isinstance(cc, list):
        raise StateValidationError("'completed_comparisons' must be a list")
    for idx, comp in enumerate(cc):
        if (
            not isinstance(comp, (list, tuple))
            or len(comp) != 2
            or not all(isinstance(c, str) for c in comp)
        ):
            raise StateValidationError(
                f"'completed_comparisons[{idx}]' must be a 2-element list of strings"
            )

    # --- history (optional) ---
    history = state.get("history", [])
    if not isinstance(history, list):
        raise StateValidationError("'history' must be a list")
    for idx, entry in enumerate(history):
        if isinstance(entry, dict):
            for key in ("item_a", "item_b"):
                if key not in entry:
                    raise StateValidationError(
                        f"history[{idx}] missing required key '{key}'"
                    )


def read_input(
    data_input: str, as_dataframe: Optional[bool] = None
) -> Union[List[Dict[str, str]], "pd.DataFrame"]:
    """
    Reads .csv into a list of dicts, or splits a comma-separated string into rows.
    Returns list of dicts with 'Item' and optionally 'Score' keys, or a DataFrame
    when as_dataframe is True and pandas is installed.
    """
    if as_dataframe is None:
        as_dataframe = pd is not None
    if as_dataframe and pd is None:
        raise ImportError("pandas is required for as_dataframe=True")

    if os.path.exists(data_input) and data_input.endswith(".csv"):
        with open(data_input, newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            return pd.DataFrame(columns=["Item"]) if as_dataframe else []

        header = rows[0]
        if "Item" in header:
            item_idx = header.index("Item")
            score_idx = header.index("Score") if "Score" in header else None
            data_rows = rows[1:]
        else:
            item_idx = 0
            score_idx = 1 if max(len(row) for row in rows) > 1 else None
            data_rows = rows

        output: List[Dict[str, str]] = []
        for row in data_rows:
            item = row[item_idx] if item_idx < len(row) else ""
            entry = {"Item": item}
            if score_idx is not None and score_idx < len(row):
                entry["Score"] = row[score_idx]
            output.append(entry)
        return pd.DataFrame(output) if as_dataframe else output

    # Handle comma-separated string input
    items = [item.strip() for item in data_input.split(",")]
    output = [{"Item": item} for item in items]
    return pd.DataFrame(output) if as_dataframe else output


def parse_input(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[str], Optional[Dict[str, float]]]:
    """
    Parse the input rows to separate items and scores.
    Expects mappings with 'Item' and optionally 'Score' keys.
    """
    if hasattr(rows, "columns"):
        columns = list(rows.columns)
        if "Item" not in columns:
            return [], None
        items = list(rows["Item"])
        if "Score" in columns:
            scores = {item: float(score) for item, score in zip(items, rows["Score"])}
        else:
            scores = None
        return items, scores

    items = [str(row.get("Item", "")) for row in rows]
    if any("Score" in row for row in rows):
        scores = {
            str(row.get("Item", "")): float(row["Score"])
            for row in rows
            if "Score" in row and row.get("Item")
        }
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
    if num_items == 0:
        return {}

    cutoff_positions = [int(c * num_items) for c in quantile_cutoffs]
    quantiles = {}
    quantile_label = 1
    next_cutoff_idx = 1

    for i, (key, _) in enumerate(sorted_ranks.items()):
        while next_cutoff_idx < len(cutoff_positions) and i >= cutoff_positions[next_cutoff_idx]:
            quantile_label += 1
            next_cutoff_idx += 1
        quantiles[key] = quantile_label

    return quantiles


def assign_levels(
    sorted_ranks: Mapping[Any, float], num_levels: int
) -> Dict[Union[int, str], int]:
    total_items = len(sorted_ranks)
    if total_items == 0:
        return {}
    items_per_level = total_items // num_levels
    remainder = total_items % num_levels
    level_sizes = [
        items_per_level + (1 if idx < remainder else 0) for idx in range(num_levels)
    ]
    levels = {}
    current_level = num_levels
    count = 0
    size_idx = 0
    for key, _ in sorted_ranks.items():
        levels[key] = current_level
        count += 1
        if count >= level_sizes[size_idx]:
            size_idx += 1
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
        self.completed_comparisons: set[tuple[str, str]] = set()
        self.history: List[
            Tuple[
                Any,
                Any,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                int,
                set[tuple[str, str]],
            ]
        ] = []

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
        # Save state before update (include counts so undo is consistent)
        self.history.append(
            (
                item_a,
                item_b,
                self.strengths.copy(),
                self.comparison_matrix.copy(),
                self.win_matrix.copy(),
                self.iteration_count,
                self.completed_comparisons.copy(),
            )
        )

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

        (
            item_a,
            item_b,
            previous_strengths,
            previous_comparison_matrix,
            previous_win_matrix,
            previous_iteration_count,
            previous_completed_comparisons,
        ) = self.history.pop()
        
        # Restore previous state
        self.strengths = previous_strengths
        self.comparison_matrix = previous_comparison_matrix
        self.win_matrix = previous_win_matrix
        self.iteration_count = previous_iteration_count
        self.completed_comparisons = previous_completed_comparisons
        
        print(f"Undid comparison between '{item_a}' and '{item_b}'")

    def get_comparison_key(
        self, item_a: Union[int, str], item_b: Union[int, str]
    ) -> tuple:
        """Create a consistent key for a comparison regardless of order"""
        return tuple(sorted([str(item_a), str(item_b)]))

    def _completed_from_matrix(self, comparison_matrix: np.ndarray) -> set[tuple[str, str]]:
        completed: set[tuple[str, str]] = set()
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                if comparison_matrix[i, j] > 0:
                    completed.add(self.get_comparison_key(self.idx_to_item[i], self.idx_to_item[j]))
        return completed

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

    def compute_fisher_information_matrix(self) -> np.ndarray:
        """Compute the Fisher Information Matrix for parameter uncertainty"""
        if np.sum(self.comparison_matrix) == 0:
            return np.eye(self.n_items) * 1e-6  # Small diagonal matrix for no comparisons
            
        # Fisher Information Matrix: F_ij = Σ_k n_ik * p_ik * (1-p_ik) * (δ_ij - δ_kj)
        # where δ_ij is Kronecker delta
        fim = np.zeros((self.n_items, self.n_items))
        
        for i in range(self.n_items):
            for j in range(self.n_items):
                for k in range(self.n_items):
                    if i != k and self.comparison_matrix[i, k] > 0:
                        p_ik = self.bradley_terry_prob(self.strengths[i], self.strengths[k])
                        n_ik = self.comparison_matrix[i, k]
                        
                        if i == j:  # Diagonal elements
                            fim[i, j] += n_ik * p_ik * (1 - p_ik)
                        elif j == k:  # Off-diagonal elements  
                            fim[i, j] -= n_ik * p_ik * (1 - p_ik)
        
        return fim

    def get_parameter_standard_errors(self) -> np.ndarray:
        """Get standard errors for strength parameters using Fisher Information Matrix"""
        fim = self.compute_fisher_information_matrix()
        
        # Handle constraint that sum of parameters = 0
        # Remove one parameter (last one) and compute inverse
        try:
            fim_reduced = fim[:-1, :-1]  # Remove last row and column
            if np.linalg.det(fim_reduced) < 1e-10:
                # Matrix is near-singular, return high uncertainty
                return np.full(self.n_items, 1.0)
                
            inv_fim = np.linalg.inv(fim_reduced)
            se_reduced = np.sqrt(np.diag(inv_fim))
            
            # Add back the constrained parameter (its SE is computed from others)
            # For zero-sum constraint: var(β_n) = var(sum of other β_i)
            se_last = np.sqrt(np.sum(inv_fim))
            se_full = np.append(se_reduced, se_last)
            
            return se_full
            
        except (np.linalg.LinAlgError, ValueError):
            # If matrix inversion fails, return high uncertainty
            return np.full(self.n_items, 1.0)

    def get_uncertainty(self, item: Union[int, str]) -> float:
        """Compute uncertainty for an item using proper Fisher Information Matrix approach"""
        i = self.item_to_idx[item]
        
        if np.sum(self.comparison_matrix) == 0:
            return 1.0  # Maximum uncertainty with no comparisons
            
        # Get standard error from Fisher Information Matrix
        standard_errors = self.get_parameter_standard_errors()
        se = standard_errors[i]
        
        # Convert standard error to 0-1 uncertainty scale
        # Using a sigmoid-like transformation: uncertainty = se / (1 + se)
        uncertainty = se / (1 + se)
        return float(np.clip(uncertainty, 0.0, 1.0))

    def get_confidence_intervals(self, alpha: float = 0.05) -> Dict[Union[int, str], Tuple[float, float]]:
        """Get confidence intervals for strength parameters"""
        standard_errors = self.get_parameter_standard_errors()
        z_score = 1.96  # For 95% confidence interval (alpha=0.05)
        
        if alpha != 0.05:
            from scipy.stats import norm
            z_score = norm.ppf(1 - alpha/2)
        
        confidence_intervals = {}
        for i, item in enumerate(self.items):
            se = standard_errors[i]
            strength = self.strengths[i]
            ci_lower = strength - z_score * se
            ci_upper = strength + z_score * se
            confidence_intervals[item] = (float(ci_lower), float(ci_upper))
            
        return confidence_intervals

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
            "history": [
                {
                    "item_a": a,
                    "item_b": b,
                    "strengths": s.tolist(),
                    "comparison_matrix": cm.tolist(),
                    "win_matrix": wm.tolist(),
                    "iteration_count": it,
                    "completed_comparisons": [list(comp) for comp in completed],
                }
                for a, b, s, cm, wm, it, completed in self.history
            ],
            "iteration_count": self.iteration_count,
            "completed_comparisons": [list(comp) for comp in self.completed_comparisons],
        }
        with open(filename, "w") as f:
            json.dump(state, f)

    def load_state(self, filename: str) -> None:
        """Load state from file"""
        with open(filename, "r") as f:
            state = json.load(f)
        _validate_state(state)
        self.items = state["items"]
        self.item_to_idx = {item: i for i, item in enumerate(self.items)}
        self.idx_to_item = {i: item for i, item in enumerate(self.items)}
        self.n_items = len(self.items)
        self.strengths = np.array(state["strengths"])
        self.comparison_matrix = np.array(state["comparison_matrix"])
        self.win_matrix = np.array(state["win_matrix"])
        self.history = []
        for idx, entry in enumerate(state.get("history", [])):
            if isinstance(entry, dict):
                item_a = entry.get("item_a")
                item_b = entry.get("item_b")
                strengths = np.array(entry.get("strengths", []))
                comparison_matrix = np.array(entry.get("comparison_matrix", []))
                win_matrix = np.array(entry.get("win_matrix", []))
                iteration_count = entry.get("iteration_count", idx)
                completed_raw = entry.get("completed_comparisons")
                if completed_raw is None:
                    completed = self._completed_from_matrix(comparison_matrix)
                else:
                    completed = {tuple(comp) for comp in completed_raw}
                self.history.append(
                    (
                        item_a,
                        item_b,
                        strengths,
                        comparison_matrix,
                        win_matrix,
                        iteration_count,
                        completed,
                    )
                )
            else:
                item_a, item_b, strengths, comparison_matrix, win_matrix = entry
                comparison_matrix = np.array(comparison_matrix)
                completed = self._completed_from_matrix(comparison_matrix)
                self.history.append(
                    (
                        item_a,
                        item_b,
                        np.array(strengths),
                        comparison_matrix,
                        np.array(win_matrix),
                        idx,
                        completed,
                    )
                )
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

    def export_rankings(
        self, format: str = "csv", as_dataframe: Optional[bool] = None
    ) -> Union[str, Dict, List[Dict[str, float]], "pd.DataFrame"]:
        """Export rankings in various formats; CSV can be a DataFrame or list."""
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
            if as_dataframe is None:
                as_dataframe = pd is not None
            if as_dataframe and pd is None:
                raise ImportError("pandas is required for as_dataframe=True")
            rows = [
                {"Item": item, "Rank": rank, "Confidence": confidences[item]}
                for item, rank in sorted_items
            ]
            return pd.DataFrame(rows) if as_dataframe else rows
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
