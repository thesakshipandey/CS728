Steps to access embeddings
    a. Each word has a token ID mentioned in token_ids.json
    b. Using those token IDs one can access the word embeddings
    c. Here, token = words.
    d. Google Drive Link for the embeddings: https://drive.google.com/drive/folders/1gofuYpY_Lqm3s-Y5Gw6uJqNn3alRubas?usp=drive_link

Steps to run SVD:
    a. Execute svd.py
    b. Change top_k for d (line 20)
    c. Change task2 flag to False if td-idf embeddings are required (line 21)
    d. Embeddings created by us for svd are available in the above drive link only under svd_embeddings
    e. the embeddings on the drive are in the format of dict with word as key and embeddings as the value.