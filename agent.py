import random
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class BaseAgent:
    def __init__(self, vocab, model, epsilon=0.2):
        self.vocab = vocab
        self.model = model
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.tested_words = set()
        self.state = []

    def observe(self, guess, reward):
        self.state.append((guess, reward))

    def act(self):
        raise NotImplementedError("Subclass must implement act().")


class SimilarityAgent(BaseAgent):
    def act(self):
        remaining = [w for w in self.vocab if w not in self.tested_words]
        if not remaining:
            return random.choice(self.vocab)
        
        if not self.state or random.random() < self.epsilon:
            guess = random.choice(remaining)
        else:
            best_guess = max(self.state, key=lambda x: x[1])[0]
            guess = max(remaining, key= lambda w:cosine_similarity(
                self.model.get_word_vector(w),
                self.model.get_word_vector(best_guess)
            ))

        self.tested_words.add(guess)
        return guess
    
class CentroidAgent(BaseAgent):
    def act(self):
        remaining = [w for w in self.vocab if w not in self.tested_words]

        if not remaining:
            return random.choice(remaining)
        
        top_k = sorted(self.state, key=lambda x: x[1], reverse=True)[:3]
        if not top_k:
            guess = random.choice(remaining)
        else:
            mean_vec = np.mean([self.model.get_word_vector(w) for w, _ in top_k], axis=0)
            guess = max(remaining, key=lambda w: cosine_similarity(
                self.model.get_word_vector(w), mean_vec
            ))

        self.tested_words.add(guess)
        return guess

class StochasticSimilarityAgent(BaseAgent):
    def __init__(self, vocab, model, top_k=500 ):
        super().__init__(vocab, model)
        self.top_k = top_k

    def act(self):
        remaining = [w for w in self.vocab if w not in self.tested_words]
        if not remaining:
            return random.choice(self.vocab)

        if not self.state:
            guess = random.choice(remaining)
        else:
            last_guess = self.state[-1][0]
            similar = self.model.get_nearest_neighbors(last_guess, k=self.top_k)
            words = [word for _, word in similar if word in remaining]
            guess = random.choice(words) if words else random.choice(remaining)

        self.tested_words.add(guess)
        return guess

class EnzoAgent(BaseAgent):
    def __init__(self, vocab, model, randoms=5):
        super().__init__(vocab, model)
        self.n_random = randoms  # nombre d'explorations aléatoires initiales

    def act(self):
        remaining = [w for w in self.vocab if w not in self.tested_words]
        if not remaining:
            guess = random.choice(self.vocab)
            self.tested_words.add(guess)
            return guess

        # Phase d'exploration aléatoire
        if len(self.state) < self.n_random:
            guess = random.choice(remaining)
            self.tested_words.add(guess)
            return guess

        # Phase d'exploitation : moyenne des 3 meilleurs
        top_k = sorted(self.state, key=lambda x: x[1], reverse=True)[:3]
        if not top_k:
            guess = random.choice(remaining)
        else:
            mean_vec = np.mean([self.model.get_word_vector(w) for w, _ in top_k], axis=0)
            guess = max(remaining, key=lambda w: cosine_similarity(
                self.model.get_word_vector(w), mean_vec
            ))

        self.tested_words.add(guess)
        return guess


class ExplorerAgent(BaseAgent):
    def __init__(self, vocab, model, exploration_steps=5):
        super().__init__(vocab, model)
        self.exploration_steps = exploration_steps

    def act(self):
        remaining = [w for w in self.vocab if w not in self.tested_words]
        if not remaining:
            return random.choice(self.vocab)

        if len(self.state) < self.exploration_steps:
            if not self.state:
                guess = random.choice(remaining)
            else:
                tested_vecs = [self.model.get_word_vector(w) for w, _ in self.state]

                def avg_similarity(word):
                    vec = self.model.get_word_vector(word)
                    similarities = [cosine_similarity(vec, v) for v in tested_vecs]
                    return np.mean(similarities)

                guess = min(remaining, key=avg_similarity)
        else:
            # Après exploration : stratégie basique (au choix)
            top_k = sorted(self.state, key=lambda x: x[1], reverse=True)[:3]
            mean_vec = np.mean([self.model.get_word_vector(w) for w, _ in top_k], axis=0)
            guess = max(remaining, key=lambda w: cosine_similarity(
                self.model.get_word_vector(w), mean_vec
            ))

        self.tested_words.add(guess)
        return guess


class Explorer2Agent(BaseAgent):
    def __init__(self, vocab, model, exploration_steps=5):
        super().__init__(vocab, model)
        self.exploration_steps = exploration_steps

    def act(self):
        remaining = [w for w in self.vocab if w not in self.tested_words]
        if not remaining:
            return random.choice(self.vocab)

        if len(self.state) < self.exploration_steps:
            if not self.state:
                guess = random.choice(remaining)
            else:
                tested_vecs = [self.model.get_word_vector(w) for w, _ in self.state]

                def min_max_similarity(word):
                    vec = self.model.get_word_vector(word)
                    return max(cosine_similarity(vec, v) for v in tested_vecs)

                guess = min(remaining, key=min_max_similarity)

        else:
            # Après exploration : stratégie basique (au choix)
            top_k = sorted(self.state, key=lambda x: x[1], reverse=True)[:3]
            mean_vec = np.mean([self.model.get_word_vector(w) for w, _ in top_k], axis=0)
            guess = max(remaining, key=lambda w: cosine_similarity(
                self.model.get_word_vector(w), mean_vec
            ))

        self.tested_words.add(guess)
        return guess
    
class WeightedExplorerAgent(BaseAgent):
    def __init__(self, vocab, model, exploration_steps=5, top_k=5):
        super().__init__(vocab, model)
        self.exploration_steps = exploration_steps
        self.top_k = top_k

    def act(self):
        remaining = [w for w in self.vocab if w not in self.tested_words]
        if not remaining:
            guess = random.choice(self.vocab)
        
        elif len(self.state) < self.exploration_steps:
            # Phase d'exploration
            if not self.state:
                guess = random.choice(remaining)
            else:
                tested_vecs = [self.model.get_word_vector(w) for w, _ in self.state]

                def avg_similarity(word):
                    vec = self.model.get_word_vector(word)
                    similarities = [cosine_similarity(vec, v) for v in tested_vecs]
                    return np.mean(similarities)

                guess = min(remaining, key=avg_similarity)
        
        else:
            # Phase d'exploitation
            top_k = sorted(self.state, key=lambda x: x[1], reverse=True)[:self.top_k]
            words = [w for w, _ in top_k]
            weights = np.array([score for _, score in top_k])

            # Éviter les poids nuls
            weights = np.maximum(weights, 1e-3)
            weights /= weights.sum()

            vecs = np.array([self.model.get_word_vector(w) for w in words])
            weighted_centroid = np.average(vecs, axis=0, weights=weights)

            guess = max(remaining, key=lambda w: cosine_similarity(
                self.model.get_word_vector(w), weighted_centroid
            ))

        self.tested_words.add(guess)
        return guess
