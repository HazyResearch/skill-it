"""Utility functions."""

import numpy as np


def get_weights_from_graph(args):
    if args.graph is not None:
        k = int(np.sqrt(len(args.graph))) # graph is list 
        graph = np.array(args.graph).reshape((k, k)).astype(float)
        for i in range(k):
            for j in range(k):
                if i != j and graph[i, j] == 1:
                    graph[i, j] = 0.5   
    elif args.graph_path is not None:
        graph = np.load(args.graph_path).astype(float)
        k = len(graph)
        for i in range(k):
            for j in range(len(graph[0])):
                if i != j and graph[i, j] == 1:
                    graph[i, j] = 0.5
                if i == j:
                    graph[i, j] = 1
    
    if args.target_mask is not None:
        target_mask = np.array([int(i) for i in args.target_mask])
        target_idxs = np.where(target_mask == 1)[0]
        graph = graph[:, target_idxs]
    
    weights_init = np.ones(k)

    if args.ni_test:
        loss_init = np.ones(graph.shape[1])
    elif args.target_mask is not None:
        loss_init = np.ones(len(target_idxs))
    else:
        loss_init = weights_init

    weights = np.multiply(weights_init, np.exp(args.eta * graph.dot(loss_init)))
    
    weights /= sum(weights)
    return weights

def get_filter_skills(slice_input, exclude_slice=None, n_slices=None):
    """
        Process the skills to sample/filter from, as well as their associated scores if any.
        
        Args:
        - slice_input: path to skills file, skill itself, or list of skills
        - exclude_slice: if slice_input is a list of multiple skill, this will make us filter/sample from slice_input - exclude_slice.
    """ 
    slice_scores = None    
    if slice_input is None:
        return None, None 
    elif len(slice_input) == 1 and ".txt" in slice_input[0]:
        slice_input = slice_input[0] 
        with open(slice_input, "r") as f:
            lines = f.read() # format per line: slice_name score(optional)
            slice_info = lines.split("\n")[:-1]
            slices = np.array([s.split(" ")[0] for s in slice_info])
            if len(slice_info[0].split(" ")) > 1:
                slice_scores = np.array([float(s.split(" ")[-1]) for s in slice_info])
    elif n_slices is not None :
        if len(slice_input) <= n_slices:
            slices = np.array(slice_input)
        elif len(slice_input) % n_slices != 0:
            raise ValueError(f"Length of slice list ({slice_input}) is not divisible by n_slices ({n_slices})")
        else:
            slices = np.array(slice_input).reshape((n_slices, -1))
    else:
        slices = np.array(slice_input)
        
    if slice_scores is None:
        slice_scores = np.ones(len(slices))
    
    if exclude_slice is not None:
        assert (exclude_slice in slices and len(slices) > 1)
        i = np.where(slices == exclude_slice)[0]
        slices = slices.delete(i)
        slice_scores = slice_scores.delete(i)
       
    return slices, slice_scores

