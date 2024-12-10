from functools import partial

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute 
from transformers import AutoModelForCausalLM, AutoTokenizer

def collate_EAP(xs):
    # breakpoint()
    clean, corrupted, labels, counterfactual_labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = tuple([(labels[i], counterfactual_labels[i]) for i in range(0, len(labels))])
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['clean'], row['corrupted'], row['label'], row['counterfactual_label']
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

# NOTE - modified for closing paren problem
def get_prob_diff(tokenizer: PreTrainedTokenizer):
    def prob_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
        # breakpoint()
        logits = get_logit_positions(logits, input_length)
        probs = torch.softmax(logits, dim=-1)#[:, year_indices]
        # breakpoint()
        results = []
        for prob, (label, counterfactual_label) in zip(probs, labels):
            results.append(prob[label] - prob[counterfactual_label])

        print(results)

        results = torch.stack(results)
        if loss:
            results = -results
        if mean: 
            results = results.mean()
        return results
    return prob_diff

def kl_div(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=True):
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    probs = torch.softmax(logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    results = kl_div(probs.log(), clean_probs.log(), log_target=True, reduction='none').mean(-1)
    return results.mean() if mean else results

cache_dir = "/projects/ziyuyao/codellama/codellama-7b/"
model_name = "codellama/CodeLlama-7b-hf"

# model_name = 'gpt2-small'
# model = HookedTransformer.from_pretrained(model_name, device='cuda')

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir)
model = HookedTransformer.from_pretrained(model_name, hf_model=model, tokenizer=tokenizer, device='cuda', dtype='float16')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

# not entirely sure why this frees up gpu memory, but it frees up about 1gb
# my best guess is that it has something to do with passing a hf_model into
# from_pretrained
torch.cuda.empty_cache()

# ds = EAPDataset('greater_than_data.csv')
ds = EAPDataset('correct_paren_data.csv')
# dataloader = ds.to_dataloader(120)
dataloader = ds.to_dataloader(1)
prob_diff = get_prob_diff(model.tokenizer)

## EAP-IG vanilla
# Instantiate a graph with a model
g = Graph.from_model(model)

# # Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g, dataloader, partial(prob_diff, loss=True, mean=True), method='EAP-IG', ig_steps=5)

# g.apply_topn(200, absolute=True)
g.apply_greedy(4000)

print(g.count_included_nodes(), g.count_included_edges())

g.prune_dead_nodes()
g.to_json('graph.json')

# gz = g.to_graphviz()
# gz.draw(f'graph.png', prog='dot')

baseline = evaluate_baseline(model, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
results = evaluate_graph(model, g, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results}")
print(g.count_included_nodes(), g.count_included_edges())
