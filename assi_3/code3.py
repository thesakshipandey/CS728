import torch
from tqdm import tqdm
from utils import PromptUtils
import random 
import numpy as np

def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
    # TODO 3: Head selection
    """
    Identify a subset of attention heads that are most useful for retrieving the correct tool.

    Requirements:
    - Use the same prompt structure as Part-2
    - Use attention patterns(query -> tool) to score heads
    - Aggregate signals across training queries
    - Return "max_heads" heads as (layer, head)

    Notes:
    - You must construct prompts and extract attentions inside this function
    - Avoid hardcoding specific queries or tools
    """

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # accumulate scores per head
    head_scores = torch.zeros(num_layers, num_heads, device=device)

    for qix in tqdm(range(len(train_queries))):

        sample = train_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        tool_ids = list(tools.keys())
        random.shuffle(tool_ids)
        putils = PromptUtils(
        tokenizer=tokenizer, 
        doc_ids=tool_ids, 
        dict_all_docs=tools,
        )
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v:k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)
        
        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        input_ids = inputs.input_ids[0]

        with torch.no_grad():
            attentions = model(**inputs).attentions 

        # Add your head scoring logic after this line
        gold_tool_id = map_docname_id[gold_tool_name]
        q_start = max([end for start, end in item_spans])
        q_end = len(input_ids)

        for l in range(num_layers):
            for h in range(num_heads):
                head_attn = attentions[l][0, h, q_start:q_end, :]
                head_attn_q = head_attn.sum(dim=0)
                
                tool_scores = []
                for d_start, d_end in item_spans:
                    score = head_attn_q[d_start:d_end].sum().item()
                    tool_scores.append(score)
                
                ranked_tools = np.argsort(tool_scores)[::-1]
                if ranked_tools[0] == gold_tool_id:
                    head_scores[l, h] += 1.0

    # TODO: select top heads
    selected_heads = []
    
    flat_scores = head_scores.flatten()
    topk_indices = torch.topk(flat_scores, max_heads).indices
    
    for idx in topk_indices:
        l = (idx // num_heads).item()
        h = (idx % num_heads).item()
        selected_heads.append((l, h))

    # example expected format:
    # [(layer1, head3), (layer5, head10), ...]
    assert len(selected_heads) == max_heads
    return selected_heads
