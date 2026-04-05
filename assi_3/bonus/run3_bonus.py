"""
Bonus: Alternative head selection strategies for Part 3.

Runs all strategies for K = [10, 20, 30] with a single Phase 1 pass.

Group A — Score-function variants (independent per-head scoring, different signal):
  reciprocal_rank  : score += 1/(rank+1); partial credit for near-misses
  margin           : score += gold_score - best_non_gold_score
  avg_attention    : score += mean length-normalised attention on gold tool

Group B — Structurally different selection methods:
  greedy           : iteratively add the head that most improves train Recall@1
  entropy          : pick heads with lowest mean attention entropy (unsupervised)
  consistency      : pick heads most consistently in top-10 across training queries

Usage (run from CS728_PA3/):
    python bonus/run3_bonus.py --model <path-to-model>
"""

import os
import sys
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ensure CS728_PA3/ is importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

from CS728_PA3.utils import load_model_tokenizer, PromptUtils, get_queries_and_items


# ── Seed ─────────────────────────────────────────────────────────────────────

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Phase 1: cache signals (one forward pass per training query) ──────────────

def cache_training_signals(train_queries, model, tokenizer, tools, device):
    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads

    putils0  = PromptUtils(tokenizer=tokenizer, doc_ids=list(tools.keys()),
                           dict_all_docs=tools)
    num_docs = len(putils0.doc_spans)
    N        = len(train_queries)

    cached_scores  = np.zeros((N, num_layers, num_heads, num_docs), dtype=np.float32)
    cached_entropy = np.zeros((N, num_layers, num_heads),           dtype=np.float32)
    gold_ids       = np.zeros(N, dtype=np.int32)

    print(f"[Phase 1] Caching training signals ({N} queries)...")
    for qi, sample in enumerate(tqdm(train_queries)):
        tool_ids = list(tools.keys())
        random.shuffle(tool_ids)
        putils   = PromptUtils(tokenizer=tokenizer, doc_ids=tool_ids,
                               dict_all_docs=tools)
        item_spans   = putils.doc_spans
        gold_ids[qi] = putils.dict_doc_name_id[sample["gold_tool_name"]]

        prompt = putils.create_prompt(query=sample["text"])
        inputs = tokenizer(prompt, return_tensors="pt",
                           add_special_tokens=False).to(device)
        q_start = max(e for _, e in item_spans)
        q_end   = len(inputs.input_ids[0])

        with torch.no_grad():
            attentions = model(**inputs).attentions

        for l in range(num_layers):
            attn_q = attentions[l][0, :, q_start:q_end, :].sum(dim=1)  # [H, seq]
            for i, (s, e) in enumerate(item_spans):
                cached_scores[qi, l, :, i] = (
                    attn_q[:, s:e].sum(dim=1).cpu().numpy() / max(e - s, 1)
                )
            del attn_q

        tool_start = item_spans[0][0]
        tool_end   = item_spans[-1][1]
        for l in range(num_layers):
            a = attentions[l][0, :, q_start:q_end, tool_start:tool_end].float()
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-12)
            cached_entropy[qi, l] = (
                -(a * (a + 1e-12).log()).sum(dim=-1).mean(dim=-1).cpu().numpy()
            )
            del a

    return cached_scores, cached_entropy, gold_ids, num_layers, num_heads


# ── Group A: Score-function variants ─────────────────────────────────────────

def select_reciprocal_rank(cached_scores, gold_ids, K, num_heads):
    """Score each head by sum of 1/(rank+1) of gold tool across training queries."""
    NL, NH = cached_scores.shape[1], cached_scores.shape[2]
    acc = np.zeros(NL * NH, dtype=np.float64)
    for q, gold in enumerate(gold_ids):
        for l in range(NL):
            ts   = cached_scores[q, l, :, :]
            rank = (ts > ts[:, gold:gold+1]).sum(axis=1)
            acc[l*NH:(l+1)*NH] += 1.0 / (rank + 1)
    topk = np.argsort(acc)[::-1][:K]
    return [(int(i // NH), int(i % NH)) for i in topk]


def select_margin(cached_scores, gold_ids, K, num_heads):
    """Score each head by sum of (gold_score - best_non_gold_score)."""
    NL, NH = cached_scores.shape[1], cached_scores.shape[2]
    acc = np.zeros(NL * NH, dtype=np.float64)
    for q, gold in enumerate(gold_ids):
        for l in range(NL):
            ts        = cached_scores[q, l, :, :]
            gold_s    = ts[:, gold]
            mask      = np.ones(ts.shape[1], dtype=bool)
            mask[gold] = False
            best_non  = ts[:, mask].max(axis=1)
            acc[l*NH:(l+1)*NH] += gold_s - best_non
    topk = np.argsort(acc)[::-1][:K]
    return [(int(i // NH), int(i % NH)) for i in topk]


def select_avg_attention(cached_scores, gold_ids, K, num_heads):
    """Score each head by mean attention it places on the gold tool."""
    NL, NH = cached_scores.shape[1], cached_scores.shape[2]
    acc = np.zeros(NL * NH, dtype=np.float64)
    for q, gold in enumerate(gold_ids):
        for l in range(NL):
            acc[l*NH:(l+1)*NH] += cached_scores[q, l, :, gold]
    topk = np.argsort(acc)[::-1][:K]
    return [(int(i // NH), int(i % NH)) for i in topk]


# ── Group B: Structurally different strategies ────────────────────────────────

def select_greedy(cached_scores, gold_ids, K, num_heads):
    """Iteratively add the head that most improves combined train Recall@1."""
    NQ, NL, NH, ND = cached_scores.shape
    all_heads = [(l, h) for l in range(NL) for h in range(NH)]
    selected  = []
    combined  = np.zeros((NQ, ND), dtype=np.float64)
    print("  [greedy] selecting heads...")
    for step in range(K):
        best_r, best_head = -1.0, None
        for l, h in all_heads:
            if (l, h) in selected:
                continue
            cand   = combined + cached_scores[:, l, h, :]
            recall = (cand.argmax(axis=1) == gold_ids).mean()
            if recall > best_r:
                best_r, best_head = recall, (l, h)
        l, h = best_head
        combined += cached_scores[:, l, h, :]
        selected.append(best_head)
        print(f"    step {step+1:2d}: added ({l:2d},{h:2d})  train R@1={best_r:.4f}")
    return selected


def select_entropy(cached_entropy, K, num_heads):
    """Select heads with lowest mean attention entropy (unsupervised)."""
    NL, NH   = cached_entropy.shape[1], cached_entropy.shape[2]
    mean_ent = cached_entropy.mean(axis=0).flatten()
    topk     = np.argsort(mean_ent)[:K]
    return [(int(i // NH), int(i % NH)) for i in topk]


def select_consistency(cached_scores, gold_ids, K, num_heads, top_k_heads=10):
    """Select heads that appear most consistently in top-10 across queries."""
    NQ, NL, NH, ND = cached_scores.shape
    consistency = np.zeros(NL * NH, dtype=np.int32)
    for q, gold in enumerate(gold_ids):
        rr = np.zeros(NL * NH, dtype=np.float32)
        for l in range(NL):
            ts   = cached_scores[q, l, :, :]
            rank = (ts > ts[:, gold:gold+1]).sum(axis=1)
            rr[l*NH:(l+1)*NH] = 1.0 / (rank + 1)
        for idx in np.argsort(rr)[::-1][:top_k_heads]:
            consistency[idx] += 1
    topk = np.argsort(consistency)[::-1][:K]
    return [(int(i // NH), int(i % NH)) for i in topk]


# ── Phase 2: evaluate on test set ────────────────────────────────────────────

def evaluate(model, tokenizer, test_queries, tools, selected_heads, device):
    correct_at_1 = correct_at_5 = total = 0
    for sample in tqdm(test_queries, leave=False):
        shuffled = list(tools.keys())
        random.shuffle(shuffled)
        putils       = PromptUtils(tokenizer=tokenizer, doc_ids=shuffled,
                                   dict_all_docs=tools)
        gold_tool_id = putils.dict_doc_name_id[sample["gold_tool_name"]]
        item_spans   = putils.doc_spans

        prompt = putils.create_prompt(query=sample["text"])
        inputs = tokenizer(prompt, return_tensors="pt",
                           add_special_tokens=False).to(device)
        q_start = max(e for _, e in item_spans)
        q_end   = len(inputs.input_ids[0])

        with torch.no_grad():
            attentions = model(**inputs).attentions

        doc_scores = torch.zeros(len(item_spans), device=device)
        for i, (s, e) in enumerate(item_spans):
            doc_len = max(e - s, 1)
            for l, h in selected_heads:
                doc_scores[i] += (
                    attentions[l][0, h, q_start:q_end, s:e].sum().item() / doc_len
                )

        ranked    = torch.argsort(doc_scores, descending=True)
        gold_rank = (ranked == gold_tool_id).nonzero(as_tuple=True)[0].item()
        if gold_rank == 0: correct_at_1 += 1
        if gold_rank < 5:  correct_at_5 += 1
        total += 1

    return correct_at_1 / total, correct_at_5 / total


# ── Main ─────────────────────────────────────────────────────────────────────

GROUPS = {
    "A: Score-function variants": [
        ("reciprocal_rank", "1/(rank+1); partial credit for near-misses"),
        ("margin",          "gold_score - best_non_gold; rewards confident separation"),
        ("avg_attention",   "mean attention on gold tool; rank-free signal"),
    ],
    "B: Structurally different": [
        ("greedy",       "iterative forward selection maximising train Recall@1"),
        ("entropy",      "lowest attention entropy over tools (unsupervised, no labels)"),
        ("consistency",  "heads most often in top-10 best heads across training queries"),
    ],
}

parser = argparse.ArgumentParser()
parser.add_argument("--model",         type=str, required=True)
parser.add_argument("--seed",          type=int, default=64)
parser.add_argument("--train_samples", type=int, default=200)
parser.add_argument("--k_values",      type=int, nargs="+", default=[10, 20, 30])
args = parser.parse_args()

if __name__ == "__main__":
    seed_all(args.seed)
    device = "cuda:0"

    print("Loading model...")
    tokenizer, model = load_model_tokenizer(
        model_name=args.model, device=device, dtype=torch.float16)

    train_queries, test_queries, tools = get_queries_and_items()
    train_queries = train_queries[:args.train_samples]

    cached_scores, cached_entropy, gold_ids, num_layers, num_heads = \
        cache_training_signals(train_queries, model, tokenizer, tools, device)

    results = {}

    for group_name, strategies in GROUPS.items():
        print(f"\n{'='*65}")
        print(f"Group {group_name}")
        print(f"{'='*65}")

        for strategy, desc in strategies:
            print(f"\n  Strategy: {strategy}")
            print(f"  {desc}")

            for K in args.k_values:
                print(f"\n  [K={K}] Selecting heads...")

                if strategy == "reciprocal_rank":
                    heads = select_reciprocal_rank(cached_scores, gold_ids, K, num_heads)
                elif strategy == "margin":
                    heads = select_margin(cached_scores, gold_ids, K, num_heads)
                elif strategy == "avg_attention":
                    heads = select_avg_attention(cached_scores, gold_ids, K, num_heads)
                elif strategy == "greedy":
                    heads = select_greedy(cached_scores, gold_ids, K, num_heads)
                elif strategy == "entropy":
                    heads = select_entropy(cached_entropy, K, num_heads)
                elif strategy == "consistency":
                    heads = select_consistency(cached_scores, gold_ids, K, num_heads)

                print(f"  Selected heads (K={K}): {heads}")
                print(f"  Evaluating on test set...")
                r1, r5 = evaluate(model, tokenizer, test_queries, tools, heads, device)
                results[(strategy, K)] = (r1, r5)
                print(f"  Recall@1: {r1:.4f}   Recall@5: {r5:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "-" * (22 + 16 * len(args.k_values))
    print(f"\n\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"{'Strategy':<22}" + "".join(f"   K={k:2d}          " for k in args.k_values))
    print(f"{'':22}" + "".join(f"   R@1     R@5  " for _ in args.k_values))
    print(sep)
    for group_name, strategies in GROUPS.items():
        print(f"\n  {group_name}")
        for strategy, _ in strategies:
            row = f"  {strategy:<20}"
            for K in args.k_values:
                r1, r5 = results.get((strategy, K), (float("nan"), float("nan")))
                row += f"   {r1:.4f}  {r5:.4f}"
            print(row)
