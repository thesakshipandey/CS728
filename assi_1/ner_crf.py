import re
import string
from datasets import load_dataset
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from seqeval.metrics import classification_report as seq_classification_report

# Load Dataset
dataset = load_dataset("conll2003")
tag_names = dataset["train"].features["ner_tags"].feature.names
pos_tag_names = dataset["train"].features["pos_tags"].feature.names

print("NER Tags:", tag_names)
for split in ["train", "validation", "test"]:
    print(f"  {split}: {len(dataset[split])} sentences")


# all conversions
def convert_to_tuples(example):
    return list(zip(
        example["tokens"],
        [pos_tag_names[p] for p in example["pos_tags"]],
        [tag_names[t] for t in example["ner_tags"]],
    ))

train_sents = [convert_to_tuples(ex) for ex in dataset["train"]]
val_sents   = [convert_to_tuples(ex) for ex in dataset["validation"]]
test_sents  = [convert_to_tuples(ex) for ex in dataset["test"]]

def sent2labels(sent):
    return [tok[-1] for tok in sent]

y_train = [sent2labels(s) for s in train_sents]
y_val   = [sent2labels(s) for s in val_sents]
y_test  = [sent2labels(s) for s in test_sents]


# helpers
_PUNCT = set(string.punctuation)

def word_shape(w):
    s = re.sub(r'(.)\1+', r'\1',
               ''.join('X' if c.isupper() else 'x' if c.islower()
                        else 'd' if c.isdigit() else c for c in w))
    return s

def has_any_digit(w):
    return any(c.isdigit() for c in w)

def is_punct_token(w):
    return len(w) > 0 and all(c in _PUNCT for c in w)

def get_token(sent, i):
    return sent[i][0]

def get_pos(sent, i):
    return sent[i][1] if len(sent[i]) >= 3 else ''



# FEATURE SET A: Original CRF (all features)
def add_word_features_A(feats, w, prefix):
    wl = w.lower()
    feats[f'{prefix}w.lower'] = wl
    feats[f'{prefix}w.shape'] = word_shape(w)
    feats[f'{prefix}w.len']   = len(w)

    feats[f'{prefix}is_upper'] = w.isupper()
    feats[f'{prefix}is_title'] = w.istitle()
    feats[f'{prefix}is_lower'] = w.islower()

    feats[f'{prefix}is_digit']   = w.isdigit()
    feats[f'{prefix}has_digit']  = has_any_digit(w)
    feats[f'{prefix}is_year']    = bool(re.fullmatch(r'(19|20)\d{2}', w))
    feats[f'{prefix}is_decimal'] = bool(re.fullmatch(r'\d+\.\d+', w))
    feats[f'{prefix}is_ordinal'] = bool(re.fullmatch(r'\d+(st|nd|rd|th)', wl))

    feats[f'{prefix}is_punct']       = is_punct_token(w)
    feats[f'{prefix}has_hyphen']     = '-' in w
    feats[f'{prefix}has_apostrophe'] = "'" in w
    feats[f'{prefix}has_dot']        = '.' in w
    feats[f'{prefix}has_slash']      = '/' in w
    feats[f'{prefix}is_initial']     = bool(re.fullmatch(r'[A-Za-z]\.', w))

    for k in (1, 2, 3, 4):
        if len(wl) >= k:
            feats[f'{prefix}pref{k}'] = wl[:k]
            feats[f'{prefix}suf{k}']  = wl[-k:]

    L = len(w)
    feats[f'{prefix}len<=2']  = (L <= 2)
    feats[f'{prefix}len3-5']  = (3 <= L <= 5)
    feats[f'{prefix}len6-8']  = (6 <= L <= 8)
    feats[f'{prefix}len>=9']  = (L >= 9)


def word2features_A(sent, i):
    w = get_token(sent, i)
    feats = {'bias': 1.0}
    add_word_features_A(feats, w, '0:')

    if i > 0:
        w_1 = get_token(sent, i - 1)
        add_word_features_A(feats, w_1, '-1:')
        feats['-1:wl|0:wl'] = w_1.lower() + '|' + w.lower()
    else:
        feats['BOS'] = True

    if i > 1:
        add_word_features_A(feats, get_token(sent, i - 2), '-2:')

    if i < len(sent) - 1:
        w1 = get_token(sent, i + 1)
        add_word_features_A(feats, w1, '+1:')
        feats['0:wl|+1:wl'] = w.lower() + '|' + w1.lower()
    else:
        feats['EOS'] = True

    if i < len(sent) - 2:
        add_word_features_A(feats, get_token(sent, i + 2), '+2:')

    return feats


# FEATURE SETS B & C
def add_good_feats(feats, token, prefix, pos_tag='', use_pos=False):
    tl = token.lower()
    feats[f'{prefix}w.lower']  = tl
    feats[f'{prefix}w.shape']  = word_shape(token)
    feats[f'{prefix}is_upper'] = token.isupper()
    feats[f'{prefix}is_title'] = token.istitle()
    feats[f'{prefix}is_lower'] = token.islower()
    for k in (2, 3, 4):
        if len(tl) >= k:
            feats[f'{prefix}suf{k}'] = tl[-k:]
    if use_pos and pos_tag:
        feats[f'{prefix}pos'] = pos_tag


def word2features_good(sent, i, use_pos=False):
    w = get_token(sent, i)
    feats = {'bias': 1.0}

    add_good_feats(feats, w, '0:', get_pos(sent, i), use_pos)

    if i > 0:
        w_1 = get_token(sent, i - 1)
        add_good_feats(feats, w_1, '-1:', get_pos(sent, i - 1), use_pos)
        feats['-1:wl|0:wl'] = w_1.lower() + '|' + w.lower()
    else:
        feats['BOS'] = True

    if i < len(sent) - 1:
        w1 = get_token(sent, i + 1)
        add_good_feats(feats, w1, '+1:', get_pos(sent, i + 1), use_pos)
        feats['0:wl|+1:wl'] = w.lower() + '|' + w1.lower()
    else:
        feats['EOS'] = True

    return feats


# Extract features
print("\nExtracting features...")

X_train_A = [[word2features_A(s, i) for i in range(len(s))] for s in train_sents]
X_val_A   = [[word2features_A(s, i) for i in range(len(s))] for s in val_sents]
X_test_A  = [[word2features_A(s, i) for i in range(len(s))] for s in test_sents]

X_train_B = [[word2features_good(s, i, use_pos=False) for i in range(len(s))] for s in train_sents]
X_val_B   = [[word2features_good(s, i, use_pos=False) for i in range(len(s))] for s in val_sents]
X_test_B  = [[word2features_good(s, i, use_pos=False) for i in range(len(s))] for s in test_sents]

X_train_C = [[word2features_good(s, i, use_pos=True) for i in range(len(s))] for s in train_sents]
X_val_C   = [[word2features_good(s, i, use_pos=True) for i in range(len(s))] for s in val_sents]
X_test_C  = [[word2features_good(s, i, use_pos=True) for i in range(len(s))] for s in test_sents]

print("Done.")


# Training and evaluation
def train_and_predict(X_train, X_val, X_test):
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1,
              max_iterations=100, all_possible_transitions=True)
    crf.fit(X_train, y_train)
    return crf.predict(X_val), crf.predict(X_test)


ALL_TAGS = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'O']

def evaluate(name, y_val_pred, y_test_pred):
    print("\n" + "=" * 75)
    print(f"  {name}")
    print("=" * 75)

    # tag based
    print("\n[Tag-level classification report — TEST]")
    print(flat_classification_report(y_test, y_test_pred, labels=ALL_TAGS, digits=4))

    # entity based
    print("[Entity-level (seqeval) report — TEST]")
    print(seq_classification_report(y_test, y_test_pred, digits=4))


configs = [
    ("A. Original CRF (all features)",    X_train_A, X_val_A, X_test_A),
    ("B. Good features WITHOUT POS",      X_train_B, X_val_B, X_test_B),
    ("C. Good features WITH POS",         X_train_C, X_val_C, X_test_C),
]

for name, Xtr, Xva, Xte in configs:
    print(f"\nTraining {name}...")
    yv_pred, yt_pred = train_and_predict(Xtr, Xva, Xte)
    evaluate(name, yv_pred, yt_pred)
