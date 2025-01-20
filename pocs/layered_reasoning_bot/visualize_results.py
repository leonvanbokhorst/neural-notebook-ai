"""
Results Visualization
------------------

Creates visualizations for both baseline and adaptive test results.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def load_results(results_dir: Path, is_baseline: bool = False) -> Dict[str, Any]:
    """Load test results from JSON files."""
    results = {}

    try:
        # Load all sequence results
        pattern = "baseline_results_*.json" if is_baseline else "test_results_*.json"
        for result_file in results_dir.glob(pattern):
            if "final" not in result_file.name:
                # Extract sequence name from filename
                parts = result_file.name.split("_")
                sequence_parts = []
                for i, part in enumerate(parts):
                    if part in [
                        "rapport",
                        "emotional",
                        "stress",
                        "cognitive",
                        "recovery",
                    ]:
                        sequence_parts.extend(parts[i : i + 2])
                        break
                sequence_name = "_".join(sequence_parts)

                logger.info(
                    f"Loading sequence: {sequence_name} from {result_file.name}"
                )
                with open(result_file) as f:
                    data = json.load(f)
                    # Handle different result formats
                    if is_baseline:
                        results[sequence_name] = {
                            "sequence_results": {
                                sequence_name: data["results"][sequence_name]
                            }
                        }
                    else:
                        results[sequence_name] = data

        logger.info(f"Loaded {len(results)} sequences: {', '.join(results.keys())}")
        return results
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        raise


def plot_emotional_dynamics_comparison(
    baseline_results: Dict[str, Any], adaptive_results: Dict[str, Any], output_dir: Path
) -> None:
    """Plot emotional dynamics comparison between baseline and adaptive results."""
    logger.info("Creating emotional dynamics comparison plot...")

    emotions = [
        "joy",
        "sadness",
        "anger",
        "fear",
        "trust",
        "disgust",
        "surprise",
        "anticipation",
    ]
    sequences = [
        "rapport_building",
        "emotional_probing",
        "stress_testing",
        "cognitive_dissonance",
        "recovery_patterns",
    ]

    # Create a figure with subplots for each sequence
    fig, axes = plt.subplots(len(sequences), 1, figsize=(15, 4 * len(sequences)))
    fig.suptitle(
        "Emotional Dynamics Comparison (Baseline vs Adaptive)", fontsize=16, y=0.95
    )

    # Color palette for emotions
    colors = plt.cm.tab10(np.linspace(0, 1, len(emotions)))

    for seq_idx, seq_name in enumerate(sequences):
        ax = axes[seq_idx]

        # Plot baseline results
        if seq_name in baseline_results:
            sequence_results = baseline_results[seq_name]["sequence_results"][seq_name]
            emotion_states = []
            for interaction in sequence_results:
                if (
                    "response" in interaction
                    and "emotional_state" in interaction["response"]
                ):
                    final_state = interaction["response"]["emotional_state"]["final"]
                    emotion_states.append([final_state[e] for e in emotions])

            if emotion_states:
                emotion_array = np.array(emotion_states)
                for i, emotion in enumerate(emotions):
                    ax.plot(
                        emotion_array[:, i],
                        linestyle="--",
                        color=colors[i],
                        label=f"baseline_{emotion}",
                        alpha=0.4,
                    )

        # Plot adaptive results
        if seq_name in adaptive_results:
            sequence_results = adaptive_results[seq_name]["sequence_results"][seq_name]
            emotion_states = []
            for interaction in sequence_results:
                if (
                    "response" in interaction
                    and "emotional_state" in interaction["response"]
                ):
                    final_state = interaction["response"]["emotional_state"]["final"]
                    emotion_states.append([final_state[e] for e in emotions])

            if emotion_states:
                emotion_array = np.array(emotion_states)
                for i, emotion in enumerate(emotions):
                    ax.plot(
                        emotion_array[:, i],
                        linestyle="-",
                        color=colors[i],
                        label=f"adaptive_{emotion}",
                        alpha=0.8,
                    )

        ax.set_title(f"Sequence: {seq_name.replace('_', ' ').title()}")
        ax.set_xlabel("Interaction Steps")
        ax.set_ylabel("Emotional Intensity")
        ax.grid(True, alpha=0.3)

        # Only show legend for the first subplot
        if seq_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(
        output_dir / "emotional_dynamics_comparison.png", bbox_inches="tight", dpi=300
    )
    plt.close()
    logger.info("Emotional dynamics comparison plot saved")


def plot_stress_energy_comparison(
    baseline_results: Dict[str, Any], adaptive_results: Dict[str, Any], output_dir: Path
) -> None:
    """Plot stress and energy levels comparison using subplots."""
    logger.info("Creating stress and energy levels comparison plot...")

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Stress and Energy Levels Comparison", fontsize=16, y=1.02)

    # Define sequences and metrics
    sequences = [
        "rapport_building",
        "emotional_probing",
        "stress_testing",
        "cognitive_dissonance",
        "recovery_patterns",
    ]
    metrics = ["stress_level", "energy_level"]

    # Process baseline results
    baseline_data = {metric: {seq: [] for seq in sequences} for metric in metrics}
    for seq_name in sequences:
        if seq_name in baseline_results.get("results", {}):
            sequence_results = baseline_results["results"][seq_name]
            for interaction in sequence_results:
                if "response" in interaction and "bot_state" in interaction["response"]:
                    bot_state = interaction["response"]["bot_state"]
                    baseline_data["stress_level"][seq_name].append(
                        bot_state.get("stress_level", 0)
                    )
                    baseline_data["energy_level"][seq_name].append(
                        bot_state.get("energy_level", 0)
                    )

    # Process adaptive results
    adaptive_data = {metric: {seq: [] for seq in sequences} for metric in metrics}
    for seq_name in sequences:
        if seq_name in adaptive_results.get("sequence_results", {}):
            sequence_results = adaptive_results["sequence_results"][seq_name]
            for interaction in sequence_results:
                if "response" in interaction and "bot_state" in interaction["response"]:
                    bot_state = interaction["response"]["bot_state"]
                    adaptive_data["stress_level"][seq_name].append(
                        bot_state.get("stress_level", 0)
                    )
                    adaptive_data["energy_level"][seq_name].append(
                        bot_state.get("energy_level", 0)
                    )

    # Plot stress levels
    baseline_stress = [baseline_data["stress_level"][seq] for seq in sequences]
    adaptive_stress = [adaptive_data["stress_level"][seq] for seq in sequences]

    ax1.boxplot(
        baseline_stress,
        positions=np.arange(len(sequences)) * 3,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.6),
        medianprops=dict(color="black"),
        tick_labels=[seq.replace("_", "\n") for seq in sequences],
    )
    ax1.boxplot(
        adaptive_stress,
        positions=np.arange(len(sequences)) * 3 + 1,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral", alpha=0.6),
        medianprops=dict(color="black"),
    )
    ax1.set_title("Stress Levels", pad=20)
    ax1.set_ylabel("Stress Level")
    ax1.grid(True, alpha=0.3)
    ax1.legend(["Baseline", "Adaptive"], loc="upper right")

    # Plot energy levels
    baseline_energy = [baseline_data["energy_level"][seq] for seq in sequences]
    adaptive_energy = [adaptive_data["energy_level"][seq] for seq in sequences]

    ax2.boxplot(
        baseline_energy,
        positions=np.arange(len(sequences)) * 3,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.6),
        medianprops=dict(color="black"),
        tick_labels=[seq.replace("_", "\n") for seq in sequences],
    )
    ax2.boxplot(
        adaptive_energy,
        positions=np.arange(len(sequences)) * 3 + 1,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral", alpha=0.6),
        medianprops=dict(color="black"),
    )
    ax2.set_title("Energy Levels", pad=20)
    ax2.set_ylabel("Energy Level")
    ax2.grid(True, alpha=0.3)
    ax2.legend(["Baseline", "Adaptive"], loc="upper right")

    plt.tight_layout()
    plt.savefig(
        output_dir / "stress_energy_comparison.png", bbox_inches="tight", dpi=300
    )
    plt.close()
    logger.info("Stress and energy levels comparison plot saved")


def plot_response_evaluation(
    baseline_results: Dict[str, Any], adaptive_results: Dict[str, Any], output_dir: Path
) -> None:
    """Plot response evaluation metrics comparing baseline and adaptive results."""
    plt.figure(figsize=(12, 6))

    sequences = [
        "rapport_building",
        "emotional_probing",
        "stress_testing",
        "cognitive_dissonance",
        "recovery_patterns",
    ]

    # Extract metrics
    baseline_metrics = {}
    adaptive_metrics = {}

    # Handle baseline results structure
    for sequence in sequences:
        if sequence in baseline_results.get("results", {}):
            sequence_data = baseline_results["results"][sequence]
            if sequence_data:
                last_interaction = sequence_data[-1]
                response = last_interaction.get("response", {})
                bot_state = response.get("bot_state", {})
                baseline_metrics[sequence] = {
                    "engagement": bot_state.get("energy_level", 0),
                    "response_time": 0,  # Not tracked in baseline
                }

    # Handle adaptive results structure
    for sequence in sequences:
        if sequence in adaptive_results.get("sequence_results", {}):
            sequence_data = adaptive_results["sequence_results"][sequence]
            if sequence_data:
                last_interaction = sequence_data[-1]
                analysis = last_interaction.get("analysis", {})
                metrics = analysis.get("interaction_metrics", {})
                adaptive_metrics[sequence] = {
                    "engagement": metrics.get("engagement_score", 0),
                    "response_time": metrics.get("average_response_time", 0),
                }

    # Plot metrics
    x = np.arange(len(sequences))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot engagement scores
    baseline_engagement = [
        baseline_metrics.get(seq, {}).get("engagement", 0) for seq in sequences
    ]
    adaptive_engagement = [
        adaptive_metrics.get(seq, {}).get("engagement", 0) for seq in sequences
    ]

    ax1.bar(
        x - width / 2, baseline_engagement, width, label="Baseline", color="lightblue"
    )
    ax1.bar(
        x + width / 2, adaptive_engagement, width, label="Adaptive", color="lightcoral"
    )
    ax1.set_ylabel("Engagement Score")
    ax1.set_title("Engagement Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels([seq.replace("_", "\n") for seq in sequences], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot response times
    baseline_times = [
        baseline_metrics.get(seq, {}).get("response_time", 0) for seq in sequences
    ]
    adaptive_times = [
        adaptive_metrics.get(seq, {}).get("response_time", 0) for seq in sequences
    ]

    ax2.bar(x - width / 2, baseline_times, width, label="Baseline", color="lightblue")
    ax2.bar(x + width / 2, adaptive_times, width, label="Adaptive", color="lightcoral")
    ax2.set_ylabel("Average Response Time (s)")
    ax2.set_title("Response Time Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels([seq.replace("_", "\n") for seq in sequences], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "response_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Response evaluation plot saved to {output_dir / 'response_evaluation.png'}")


def create_visualizations(results_dir: Path, baseline_dir: Path) -> None:
    """Create all visualizations comparing baseline and adaptive results."""
    try:
        # Load both baseline and adaptive results
        baseline_results = load_results(baseline_dir, is_baseline=True)
        adaptive_results = load_results(results_dir, is_baseline=False)

        # Create output directory
        output_dir = results_dir / "visualizations"
        output_dir.mkdir(exist_ok=True)

        # Create comparison visualizations
        plot_emotional_dynamics_comparison(
            baseline_results, adaptive_results, output_dir
        )
        plot_stress_energy_comparison(baseline_results, adaptive_results, output_dir)
        plot_response_evaluation(baseline_results, adaptive_results, output_dir)

        logger.info(f"Created comparison visualizations in {output_dir}")

    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        raise


if __name__ == "__main__":
    # Get the latest results directories
    results_dir = Path(__file__).parent / "test_results"
    baseline_dir = Path(__file__).parent / "baseline_results"

    if not results_dir.exists() or not baseline_dir.exists():
        print(
            f"Missing results directories:\nTest results: {results_dir}\nBaseline: {baseline_dir}"
        )
        exit(1)

    # Get the most recent directories
    result_dirs = sorted(
        results_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True
    )
    baseline_dirs = sorted(
        baseline_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True
    )

    if not result_dirs or not baseline_dirs:
        print("No result directories found")
        exit(1)

    latest_results = result_dirs[0]
    latest_baseline = baseline_dirs[0]

    print(
        f"Creating visualizations comparing:\nBaseline: {latest_baseline}\nAdaptive: {latest_results}"
    )
    create_visualizations(latest_results, latest_baseline)
