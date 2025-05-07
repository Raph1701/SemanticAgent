import json
import os
import numpy as np

def cosine_distance(vec1, vec2):
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

def collect_episode_data(agent_name, env, model):
    """Récupère les données utiles à la fin d'un épisode."""
    data = {
        "agent_name": agent_name,
        "target_word": env.secret_word,
        "guesses": [],
        "rewards": [],
        "distances": [],
        "steps": env.steps,
        "success": env.history[-1][0] == env.secret_word if env.history else False
    }

    vec_target = model.get_word_vector(env.secret_word)

    for guess, reward in env.history:
        data["guesses"].append(guess)
        data["rewards"].append(reward)
        vec_guess = model.get_word_vector(guess)
        dist = cosine_distance(vec_guess, vec_target)
        data["distances"].append(dist)

    return data

def save_results(all_episodes_data, filename="results.json"):
    """Sauvegarde les résultats dans un fichier JSON."""
    with open(filename, "w") as f:
        json.dump(all_episodes_data, f, indent=2, default=convert_np)

