# not a task specific file.
import json

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