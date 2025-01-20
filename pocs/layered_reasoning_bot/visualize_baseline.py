"""
Baseline Results Visualization
---------------------------

Creates visualizations for baseline test results to compare with adaptive version.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def load_baseline_results(results_dir: Path) -> Dict[str, Any]:
    """Load baseline results from JSON files."""
    results = {}

    try:
        # Load all sequence results
        for result_file in results_dir.glob("baseline_results_*.json"):
            if "final_baseline" not in result_file.name:
                # Extract sequence name from filename
                sequence_name = "_".join(result_file.name.split("_")[:-1])
                sequence_name = sequence_name.split("baseline_ai_")[-1]
                if sequence_name.endswith("_baseline"):
                    sequence_name = sequence_name[:-9]

                logger.info(
                    f"Loading sequence: {sequence_name} from {result_file.name}"
                )
                with open(result_file) as f:
                    results[sequence_name] = json.load(f)

        logger.info(f"Loaded {len(results)} sequences: {', '.join(results.keys())}")
        return results
    except Exception as e:
        logger.error(f"Error loading baseline results: {str(e)}")
        raise


def plot_emotional_dynamics(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot emotional dynamics across test sequences."""
    logger.info("Creating emotional dynamics plot...")
    plt.figure(figsize=(12, 6))

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
    sequence_emotions = {seq: [] for seq in results.keys()}

    for seq_name, seq_data in results.items():
        sequence_results = seq_data["results"].get(seq_name, [])
        logger.info(f"Processing {len(sequence_results)} interactions for {seq_name}")

        for interaction in sequence_results:
            if (
                "response" in interaction
                and "emotional_state" in interaction["response"]
            ):
                final_state = interaction["response"]["emotional_state"]["final"]
                sequence_emotions[seq_name].append([final_state[e] for e in emotions])

    # Plot emotional dynamics
    for seq_name, emotion_values in sequence_emotions.items():
        if emotion_values:
            emotion_array = np.array(emotion_values)
            logger.info(
                f"Plotting {len(emotion_values)} emotional states for {seq_name}"
            )
            for i, emotion in enumerate(emotions):
                plt.plot(emotion_array[:, i], label=f"{seq_name}_{emotion}", alpha=0.6)

    plt.title("Emotional Dynamics Across Test Sequences (Baseline)")
    plt.xlabel("Interaction Steps")
    plt.ylabel("Emotional Intensity")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_emotional_dynamics.png")
    plt.close()
    logger.info("Emotional dynamics plot saved")


def plot_stress_energy_levels(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot stress and energy levels across test sequences."""
    logger.info("Creating stress and energy levels plot...")
    plt.figure(figsize=(12, 6))

    stress_levels = []
    energy_levels = []
    sequence_names = []

    for seq_name, seq_data in results.items():
        sequence_results = seq_data["results"].get(seq_name, [])
        logger.info(f"Processing {len(sequence_results)} interactions for {seq_name}")

        for interaction in sequence_results:
            if "response" in interaction and "bot_state" in interaction["response"]:
                bot_state = interaction["response"]["bot_state"]
                stress_levels.append(bot_state["stress_level"])
                energy_levels.append(bot_state["energy_level"])
                sequence_names.append(seq_name)

    logger.info(
        f"Total measurements - Stress: {len(stress_levels)}, Energy: {len(energy_levels)}"
    )

    data = {
        "Sequence": sequence_names * 2,
        "Level": stress_levels + energy_levels,
        "Type": ["Stress"] * len(stress_levels) + ["Energy"] * len(energy_levels),
    }

    sns.boxplot(x="Sequence", y="Level", hue="Type", data=data)
    plt.title("Stress and Energy Levels by Test Sequence (Baseline)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_stress_energy_levels.png")
    plt.close()
    logger.info("Stress and energy levels plot saved")


def plot_response_evaluation(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot response evaluation metrics across test sequences."""
    logger.info("Creating response evaluation plot...")
    plt.figure(figsize=(12, 6))

    dimensions = [
        "coherence",
        "relevance",
        "emotional_alignment",
        "strategy_adherence",
        "engagement",
    ]
    sequence_scores = {seq: {dim: [] for dim in dimensions} for seq in results.keys()}

    for seq_name, seq_data in results.items():
        sequence_results = seq_data["results"].get(seq_name, [])
        logger.info(f"Processing {len(sequence_results)} interactions for {seq_name}")

        for interaction in sequence_results:
            if (
                "response" in interaction
                and "dimension_scores" in interaction["response"]
            ):
                scores = interaction["response"]["dimension_scores"]
                for dim in dimensions:
                    if dim in scores:
                        sequence_scores[seq_name][dim].append(scores[dim]["current"])

    # Plot average scores for each dimension
    x = np.arange(len(results))
    width = 0.15

    for i, dim in enumerate(dimensions):
        averages = [
            np.mean(sequence_scores[seq][dim]) if sequence_scores[seq][dim] else 0
            for seq in results.keys()
        ]
        logger.info(f"Average {dim} scores: {dict(zip(results.keys(), averages))}")
        plt.bar(x + i * width, averages, width, label=dim)

    plt.title("Response Evaluation Metrics by Test Sequence (Baseline)")
    plt.xlabel("Test Sequence")
    plt.ylabel("Average Score")
    plt.xticks(x + width * 2, list(results.keys()), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_response_evaluation.png")
    plt.close()
    logger.info("Response evaluation plot saved")


def create_baseline_visualizations(results_dir: Path) -> None:
    """Create all baseline visualizations."""
    try:
        # Load results
        results = load_baseline_results(results_dir)

        # Create output directory
        output_dir = results_dir / "visualizations"
        output_dir.mkdir(exist_ok=True)

        # Create visualizations
        plot_emotional_dynamics(results, output_dir)
        plot_stress_energy_levels(results, output_dir)
        plot_response_evaluation(results, output_dir)

        logger.info(f"Created baseline visualizations in {output_dir}")

    except Exception as e:
        logger.error(f"Error creating baseline visualizations: {str(e)}")
        raise


if __name__ == "__main__":
    # Get the latest baseline results directory
    baseline_dir = Path(__file__).parent / "baseline_results"
    if not baseline_dir.exists():
        print(f"No baseline results directory found at {baseline_dir}")
        exit(1)

    # Get the most recent results directory
    result_dirs = sorted(
        baseline_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True
    )
    if not result_dirs:
        print("No baseline result directories found")
        exit(1)

    latest_dir = result_dirs[0]
    print(f"Creating visualizations for results in {latest_dir}")
    create_baseline_visualizations(latest_dir)
