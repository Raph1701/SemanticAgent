from word_env import WordEnvironment
from agent import *
from save import collect_episode_data, save_results
import fasttext
import random
import numpy as np
import matplotlib.pyplot as plt

def load_model(path='cc.en.300.bin'):
    print("Loading fastText model...")
    model= fasttext.load_model(path)
    print("Loaded.")
    
    return model
    
def build_vocab(model, limit=5000):
    return [word for word in model.get_words()[:limit] if word.isalpha()]

def run_agent(agent_class, model, vocab, n_episodes=1):
    env = WordEnvironment(model, vocab=vocab, max_steps=200)
    agent = agent_class(vocab, model)
    steps_per_episode = []

    for ep in range(n_episodes):
        print(f"Episode {ep+1}/{n_episodes}:")
        state = env.reset()
        agent.reset()

        done = False
        steps = 0
        while not done:
            guess = agent.act()
            print(f"guess {steps +1}: {guess}")
            state, reward, done = env.step(guess)

            agent.observe(guess, reward)
            steps += 1
        steps_per_episode.append(steps)

    return steps_per_episode

def compare_agents():
    model = load_model()
    vocab = build_vocab(model)

    agents = {
        "SimilarityAgent": SimilarityAgent,
        "CentroidAgent": CentroidAgent,
    }

    all_results = {}

    for name, agent_class in agents.items():
        print(f"Training {name}...")
        results = run_agent(agent_class, model, vocab)
        all_results[name] = results

    for name, results in all_results.items():
        plt.plot(results, label=name)

    plt.xlabel("Episode")
    plt.ylabel("Number of Steps")
    plt.legend()
    plt.title("Agent Performance Comparison")
    plt.grid(True)
    plt.show()

def train(agent_class, max_steps=200, n_episodes=20):
    model = load_model()
    vocab = build_vocab(model)
    name  = agent_class.__name__

    env = WordEnvironment(model, vocab, max_steps=max_steps)
    agent = agent_class(vocab, model)
    all_data = []

    for episode in range(n_episodes):
        print(f"Episode {episode+1}/{n_episodes}")
        env.reset()
        agent.reset()

        done = False
        steps = 0

        while not done:
            guess = agent.act()
            state, reward, done = env.step(guess)
            agent.observe(guess, reward)
            steps += 1
            print(f"Step {steps}/200")

        episode_data = collect_episode_data(name, env, model)
        episode_data['episode'] = episode
        all_data.append(episode_data)
    save_results(all_data, filename=f"results_{name}.json")


if __name__ == "__main__":
    #compare_agents()
    train(WeightedExplorerAgent, n_episodes=20)