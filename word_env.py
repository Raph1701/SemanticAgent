import random
import numpy as np

class WordEnvironment:
    def __init__(self, model, vocab, max_steps=20):
        self.model = model
        self.vocab = vocab
        self.vocab_set = set(vocab)  # pour lookup rapide
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.secret_word = random.choice(self.vocab)
        self.steps = 0
        self.done = False
        self.history = []
        return self._get_state()
    
    def _get_state(self):
        return self.history.copy()
    
    def get_secret_word(self):
        return self.secret_word
    
    def _cosine_similarity(self, w1, w2):
        v1 = self.model.get_word_vector(w1)
        v2 = self.model.get_word_vector(w2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def step(self, guess):
        if self.done:
            raise Exception("L'épisode est terminé. Appelle reset() avant.")
        
        self.steps += 1

        if guess not in self.vocab_set:
            reward = -1.0  # mot inconnu
        else:
            similarity = self._cosine_similarity(guess, self.secret_word)
            reward = similarity

        self.history.append((guess, reward))

        if guess == self.secret_word:
            self.done = True
            reward = 1.0  # victoire
        elif self.steps >= self.max_steps:
            self.done = True  # échec

        return self._get_state(), reward, self.done
