from typing import Callable, List, Union

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import rearrange, einsum

from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode, Node

def evaluate_graph(model: HookedTransformer, graph: Graph,  clean_inputs, corrupted_inputs, labels, metric: Callable[[Tensor], Tensor], prune:bool=True):
    """
    Evaluate a circuit (i.e. a graph where only some nodes are false, probably created by calling graph.apply_threshold). You probably want to prune beforehand to make sure your circuit is valid.
    """
    if prune:
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)

    fwd_names = graph.parent_node_names()
    fwd_filter = lambda x: x in fwd_names
    
    corrupted_fwd_cache, corrupted_fwd_hooks, _ = model.get_caching_hooks(fwd_filter)
    mixed_fwd_cache, mixed_fwd_hooks, _ = model.get_caching_hooks(fwd_filter)

    nodes_in_graph = [node for node in graph.nodes.values() if node.in_graph if not isinstance(node, InputNode)]

    # For each node in the graph, construct its input (in the case of attention heads, multiple inputs) by corrupting the incoming edges that are not in the circuit.
    # We assume that the corrupted cache is filled with corresponding corrupted activations, and that the mixed cache contains the computed activations from preceding nodes in this forward pass.
    def make_input_construction_hook(node: Node, qkv=None):
        def input_construction_hook(activations, hook):
            for edge in node.parent_edges:
                if edge.qkv != qkv:
                    continue

                parent:Node = edge.parent
                if not edge.in_graph:
                    activations[edge.index] -= mixed_fwd_cache[parent.out_hook][parent.index]
                    activations[edge.index] += corrupted_fwd_cache[parent.out_hook][parent.index]
            return activations
        return input_construction_hook

    input_construction_hooks = []
    for node in nodes_in_graph:
        if isinstance(node, InputNode):
            pass
        elif isinstance(node, LogitNode) or isinstance(node, MLPNode):
            input_construction_hooks.append((node.in_hook, make_input_construction_hook(node)))
        elif isinstance(node, AttentionNode):
            for i, letter in enumerate('qkv'):
                input_construction_hooks.append((node.qkv_inputs[i], make_input_construction_hook(node, qkv=letter)))
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")
            
    # and here we actually run / evaluate the model
    results = []
    for clean, corrupted, label in tqdm(zip(clean_inputs, corrupted_inputs, labels), total=len(clean_inputs)):
        tokenized = model.tokenizer(clean, padding='longest', return_tensors='pt', add_special_tokens=True)
        input_lengths = 1 + tokenized.attention_mask.sum(1)
        with torch.inference_mode():
            with model.hooks(corrupted_fwd_hooks):
                corrupted_logits = model(corrupted)

            with model.hooks(mixed_fwd_hooks + input_construction_hooks):
                logits = model(clean)

        r = metric(logits, corrupted_logits, input_lengths, label).cpu()
        if len(r.size()) == 0:
            r = r.unsqueeze(0)
        results.append(r)

    return torch.cat(results)

def evaluate_baseline(model: HookedTransformer, clean_inputs, corrupted_inputs, labels, metric: Callable[[Tensor], Tensor]):
    results = []
    for clean, corrupted, label in tqdm(zip(clean_inputs, corrupted_inputs, labels), total=len(clean_inputs)):
        tokenized = model.tokenizer(clean, padding='longest', return_tensors='pt', add_special_tokens=True)
        input_lengths = 1 + tokenized.attention_mask.sum(1)
        with torch.inference_mode():
            corrupted_logits = model(corrupted)
            logits = model(clean)
        r = metric(logits, corrupted_logits, input_lengths, label).cpu()
        if len(r.size()) == 0:
            r = r.unsqueeze(0)
        results.append(r)

    return torch.cat(results)


def evaluate_kl(model: HookedTransformer, inputs, target_inputs):
    results = []
    for inp, target in tqdm(zip(inputs, target_inputs), total=len(inputs)):
        
        batch_size = len(inp)
        tokenized = model.tokenizer(inp, padding='longest', return_tensors='pt', add_special_tokens=True)
        input_length = 1 + tokenized.attention_mask.sum(1)
        
        with torch.inference_mode():
            target_logits = model(target)
            logits = model(inp)

        idx = torch.arange(batch_size, device=logits.device)

        logits = logits[idx, input_length - 1]
        target_logits = target_logits[idx, input_length - 1]

        logprobs = torch.log_softmax(logits, dim=-1)
        target_logprobs = torch.log_softmax(target_logits, dim=-1)

        r = torch.nn.functional.kl_div(logprobs, target_logprobs, log_target=True, reduction='mean')
        if len(r.size()) == 0:
            r = r.unsqueeze(0)
        results.append(r)

    return torch.cat(results)