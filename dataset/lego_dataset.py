import numpy as np
import random
from collections import defaultdict
import torch
import networkx as nx
from .dataset import AbstractDataset
from .utils import get_weights_from_graph, get_filter_skills

class LegoDataset(AbstractDataset):
    def __init__(
        self,
        args,
        logger,
        tokenizer,
        seed,
        sample_rule,
        n_segment,
        is_eval,
    ):
        super().__init__(args, logger, tokenizer, seed, sample_rule, is_eval)

        # number of variables in each sentence.
        assert (args.k is not None) # for LEGO, this needs to be provided to us 
        self.k = args.k
        self.logger.info(f"building LEGO dataset. Number of skills: {self.k}")

        # number of segments to divide training into
        self.n_segment = n_segment

        self.set_skills(args)
        self.set_proportions(args, args.proportions)
        
        self._set_lego_graph(args)

        self.all_vars = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
        ]
                
        self.all_sentences = defaultdict(set)
        
    def set_skills(self, args):
        skills, _ = get_filter_skills(args.slice_list, args.exclude_slice, self.k)
        
        if skills is not None and (not self.is_eval or (self.is_eval and args.filter_val_skills)):
            self.skills = np.tile(np.array(skills).astype(int), (self.n_segment, 1))
        else:
            self.skills = np.tile(
                np.arange(self.k), (self.n_segment, 1)
            ) 
        self.n_var = self.k if self.is_eval else len(self.skills[0]) # size of support for sampling distribution

    def set_proportions(self, args, proportions):
        if self.sample_rule == "stratified":
            self.proportions = np.tile(
                np.repeat(1.0 / self.n_var, self.n_var), (self.n_segment, 1)
            ) 
        elif self.sample_rule == "mixture":
            if proportions is not None:
                assert len(proportions) == self.n_var * self.n_segment, f"Length of the proportions list ({len(proportions)}) does not equal n_var ({self.n_var}) * n_segment ({self.n_segment})."
                proportions = np.array(np.split(np.array(proportions), self.n_segment))
                row_sums = proportions.sum(axis=1)
                proportions = proportions / row_sums[:, np.newaxis]
                self.proportions = proportions
            elif (args.graph is not None or args.graph_path is not None) and not args.mw:
                self.proportions = get_weights_from_graph(args)
                proportions = np.array(np.split(np.array(proportions), self.n_segment))
                row_sums = proportions.sum(axis=1)
                proportions = proportions / row_sums[:, np.newaxis]
                self.proportions = proportions
                self.logger.info(f"Setting weights from graph: {self.proportions}")
            else:
                self.proportions = np.tile(
                    np.repeat(1.0 / self.n_var, self.n_var), (self.n_segment, 1)
                ) 
        else:
            raise ValueError(f"Invalid sample rule: {self.sample_rule}")
                        
        if args.segment_proportions is not None:
            assert len(args.segment_proportions) == self.n_segment
            segment_proportions = np.array(args.segment_proportions)
            self.segment_proportions = segment_proportions / sum(
                segment_proportions
            )  # list
        else:
            self.segment_proportions = np.repeat(1.0 / self.n_segment, self.n_segment)
            
        self.logger.info(f"Proportions: {self.proportions}\nSegment proportions: {self.segment_proportions}")

        
    def get_tokenized_dataset(self, n_data=None, include_skill_idxs=False, text=False):
        if self.is_eval:
            return self._get_tokenized_val(n_data if n_data is not None else 100 * self.n_var, text)
        else:
            return self._get_tokenized_train(n_data, include_skill_idxs, text)

    def _get_tokenized_train(self, n_data, include_skill_idxs=False,  text=False):
        all_data = []
        all_skill_idxs = []
                
        segment_lengths = (self.segment_proportions * n_data).astype(int)
        segment_lengths[-1] = n_data - segment_lengths[:-1].sum()

        for skills, proportions, seg_length in zip(
            list(self.skills), list(self.proportions), segment_lengths
        ):
            if seg_length == 0:
                continue
            data, skill_idxs, _, _ = self._generate_data_segment(
                skills, proportions, seg_length, text
            )
            all_data.extend(data)
            all_skill_idxs.extend(skill_idxs)

        return LegoTorchDataset(all_data, self.is_eval, skill_idxs = all_skill_idxs if include_skill_idxs else [], text=text)

    def _get_tokenized_val(self, n_data, text=False): 
        all_data, all_skill_idxs, all_labels, output_idxs = self._generate_data_segment(self.skills[0], self.proportions[0], n_data, text)
        return (
            LegoTorchDataset(all_data, self.is_eval, all_skill_idxs, text, all_labels),
            output_idxs,
        )
        

    def _set_lego_graph(self, args):
        if args.lego_graph is None:
            self.G = nx.path_graph(self.k, create_using=nx.DiGraph)
            self.logger.info(
                f"Using default path graph. Adjacency matrix is {nx.adjacency_matrix(self.G).toarray()}"
            )
        elif args.lego_graph == "disconnected":
            # makes two path graphs
            idxs_per_component = np.array_split(np.arange(self.k), 2)
            A = np.zeros((self.k, self.k))
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    A[i, j] = (
                        1
                        if (i in idxs_per_component[0] and j in idxs_per_component[0])
                        or (i in idxs_per_component[1] and j in idxs_per_component[1])
                        else 0
                    )
            self.G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            self.logger(
                f"Using disconnected path graphs. Adjacency matrix is {nx.adjacency_matrix(self.G).toarray()}"
            )
        elif args.lego_graph == "tree":
            self.G = nx.full_rary_tree(2, self.k, create_using=nx.DiGraph)
            self.logger(
                f"Using tree. Adjacency matrix is {nx.adjacency_matrix(self.G).toarray()}"
            )
        else:
            raise NotImplementedError(f"Invalid lego_graph: {args.lego_graph}")
       


    def _generate_data_segment(self, skills, proportions, n, text=False):
        """
        Generates a segment of data.

        Args:
        - skills: list of skill indices
        - proportions: respective numpy array of mixture proportions (sums to 1)
        - n: amount of data overall to generate for the segment
        """
        data = []  # list of batches of data
        all_skill_idxs = []
        all_labels = []

        # convert proportions into amount of data
        n_per_skill = (proportions * n).astype(int)
        n_per_skill[-1] = n - n_per_skill[:-1].sum()

        for i, skill in enumerate(skills):            
            (
                batch,
                skill_idxs,
                labels,
                output_idxs,
            ) = self._generate_skill(skill, n_per_skill[i], text)
            
            data.extend(batch)
            all_skill_idxs.extend(skill_idxs)
            if self.is_eval:
                all_labels.extend(labels)

        # shuffle training data
        if not self.is_eval:
            data, all_skill_idxs = self._shuffle_lists(data, all_skill_idxs)
            data, all_skill_idxs = list(data), list(all_skill_idxs)

        return data, all_skill_idxs, all_labels, output_idxs


    def _shuffle_lists(self, *ls):
        # shuffle a tuple of lists
        l = list(zip(*ls))
        random.shuffle(l)
        return zip(*l)


    def _generate_skill(self, skill_idx, n, text=False):
        batch = []
        skill_idxs = []
        labels = []

        roots = [
            n for n, d in self.G.in_degree() if d == 0
        ]  # get root of the LEGO graph. For default graph, this is just skill 0.

        output_idxs = None # for masking
        count = 0
        while count != n:
            values = np.random.randint(
                0, 2, (self.k,)
            )  # true assignment of the k variable labels
            var_idx = tuple(
                np.random.permutation(len(self.all_vars))
            )  # shuffle the alphabet
            vars = [self.all_vars[i] for i in var_idx]  # vars = shuffled alphabet
            # generate first sentence
            clauses = []
            for root in roots:
                clauses.append(
                    "%s = val %d , " % (vars[root], values[root])
                )  # add the first "sentence" to the clause, e.g. C = val 1. This is the "root" of the tree.

            for i in set(np.arange(self.k)).difference(set(roots)):
                # go through the remaining leaf nodes
                parent = list(self.G.in_edges(i))[0][0]
                modifier = "val" if values[i] == values[parent] else "not"
                clauses.append(
                    " %s = %s %s , " % (vars[i], modifier, vars[parent])
                )  # add the remaining sentences, which are all like B = val C or B = not C
                # spaces are to help with tokenization... (double check this?)

            sent = "Input: "
            clause_idx = tuple(
                np.random.permutation(self.k)
            )  # now, shuffle the variable assignments
            sent += "".join(
                [clauses[idx] for idx in clause_idx]
            )  # create final sentence that contains shuffled sentences

            sent = sent[: len(sent) - 3]  # get part of sentence without the last " , "

            # now add outputs
            sent += ". Output:"
            sent += " %s = val " % (vars[skill_idx])

            opposite_sent = sent + "%d, " % (1 - values[skill_idx])
            sent += "%d, " % (values[skill_idx])
            
            # remove duplicate sentences
            if sent in self.all_sentences[skill_idx]:
                continue
            else:
                self.all_sentences[skill_idx].add(sent)

            if not self.is_eval:
                tokenized = self.tokenizer(
                    sent,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.context_length,
                ) 
            else:
                tokenized = self.tokenizer(sent, return_tensors="pt")
                tokenized_opposite = self.tokenizer(opposite_sent, return_tensors="pt")
                if output_idxs is None:
                    output_idxs = []
                    for index, (id_1, id_2) in enumerate(
                        zip(
                            tokenized["input_ids"][0],
                            tokenized_opposite["input_ids"][0],
                        )
                    ):
                        if id_1 != id_2:
                            output_idxs.append(index)

            batch.append(tokenized if not text else sent)
            skill_idxs.append(skill_idx)
            if self.is_eval:
                labels.append(values[skill_idx])

            count += 1
        
        return batch, skill_idxs, labels, output_idxs



class LegoTorchDataset(torch.utils.data.Dataset):
    def __init__(self, data, is_eval, skill_idxs=[], text=False, labels=None):
        self.data = data
        self.is_eval = is_eval
        self.skill_idxs = skill_idxs
        self.text = text
        self.labels = labels

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.text:
            return {"text": data, "skill_idxs": self.skill_idxs[idx]}

        if self.is_eval:
            return {
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"],
                "skill_idxs": self.skill_idxs[idx],
                "output_labels": self.labels[idx],
            }
        elif len(self.skill_idxs) > 0:
            return {
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"],
                "skill_idxs": self.skill_idxs[idx],
            }
        else:
            # train dataset
            return {
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"],
            }

    def __len__(self):
        return len(self.data)


