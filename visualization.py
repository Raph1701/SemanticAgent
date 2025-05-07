import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def plot_attempts_histograms(all_data):
    """
    Histogrammes normalisés (densité) des steps pour chaque agent sur la même figure.
    """
    plt.figure(figsize=(8, 5))
    
    for agent_name, data in all_data.items():
        steps = [ep['steps'] for ep in data]
        plt.hist(steps, bins= 5,
                 density=True, alpha=0.5, label=agent_name, edgecolor='black')

    plt.title("Step Count Distribution to Success")
    plt.xlabel("Steps")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_avg_convergence_curves(all_data):
    """
    Courbes de distance moyenne à la cible, tracées sur un même graphique.
    """
    plt.figure(figsize=(9, 5))

    for agent_name, data in all_data.items():
        max_len = max(len(ep["distances"]) for ep in data)
        curves = []
        for ep in data:
            d = ep["distances"]
            padded = d + [d[-1]] * (max_len - len(d))
            curves.append(padded)

        avg_curve = np.mean(curves, axis=0)
        plt.plot(avg_curve, label=agent_name)

    plt.xlabel("Step")
    plt.ylabel("Average Cosine Distance to Target")
    plt.title("Convergence Curves Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    result_files = [
        "results_SimilarityAgent.json",
        "results_CentroidAgent.json",
        "results_EnzoAgent.json",
        "results_ExplorerAgent.json",
        "results_Explorer2Agent.json",
        "results_WeightedExplorerAgent.json",
        # Tu peux ajouter ici d'autres fichiers de résultats
    ]

    all_data = {}
    for file in result_files:
        agent_name = os.path.splitext(os.path.basename(file))[0].replace("results_", "")
        all_data[agent_name] = load_results(file)

    plot_attempts_histograms(all_data)
    plot_avg_convergence_curves(all_data)

if __name__ == "__main__":
    main()
