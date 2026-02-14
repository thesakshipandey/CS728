# not a task specific file.
import json
import numpy as np

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def damping_function(x_ij: int, x_max: int, alpha: float):
  if x_ij < x_max:
    return (x_ij/x_max)**alpha
  else:
    return 1
  
def make_token_ids(words_in_vocab: set[str]):
    count = 0
    token_ids = {}
    for temp in words_in_vocab:
        token_ids[temp] = count
        count += 1
    with open("token_ids.json", "w") as f:
        json.dump(token_ids, f)

def get_token_ids() -> dict[str, int]:
    with open("token_ids.json", "r") as f:
        token_ids = json.load(f)

    return token_ids

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

def get_top_k_similar_words(word, word_embeddings, token_ids, k=5):
    target_vec = word_embeddings[token_ids[word]]
    similarities = []

    for other_word in word_embeddings.keys():
        if other_word == token_ids[word]:
            continue  # skip the word itself
        sim = cosine_similarity(target_vec, word_embeddings[other_word])
        similarities.append((other_word, sim))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]