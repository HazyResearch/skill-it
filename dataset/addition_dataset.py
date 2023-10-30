import numpy as np
import random
from collections import defaultdict
import torch

from .dataset import AbstractDataset

from .utils import get_weights_from_graph, get_filter_skills

class AdditionDataset(AbstractDataset):
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
        assert (args.k is not None)
        self.k = args.k
        self.logger.info(f"building addition dataset. Number of skills: {self.k}")

        # number of segments to divide training into
        self.n_segment = n_segment
        
        self.set_skills(args)
        self.set_proportions(args, args.proportions)
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
                )  # uniform sampling
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

    def get_tokenized_dataset(self, n_data=None, include_skill_idxs=False):
        if self.is_eval:
            return self._get_tokenized_val(n_data if n_data is not None else 100 * self.n_var)
        else:
            return self._get_tokenized_train(n_data, include_skill_idxs)
        
    def _get_tokenized_train(self, n_data, include_skill_idxs=False):
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
                skills, proportions, seg_length
            )
            all_data.extend(data)
            all_skill_idxs.extend(skill_idxs)

        return AdditionTorchDataset(all_data, self.is_eval, skill_idxs = all_skill_idxs if include_skill_idxs else [])

    def _get_tokenized_val(self, n_data):
        all_data = []
        all_skill_idxs = []
        all_labels = []
        
        all_data, all_skill_idxs, all_labels, output_idxs = self._generate_data_segment(
            self.skills[0], self.proportions[0], n_data
        )

        return (
            AdditionTorchDataset(all_data, self.is_eval, all_skill_idxs, all_labels),
            output_idxs,
        )
        

    def _generate_data_segment(self, skills, proportions, n):
        batch = []
        skill_idxs = []
        all_labels = []
        n_per_skill = (proportions * n).astype(int)
        n_per_skill[-1] = n - n_per_skill[:-1].sum()
        for i, skill in enumerate(skills):
            if n_per_skill[i] == 0:
                continue 
            sents, labels, output_idxs = self._generate_skill(
                skill, n_per_skill[i])
            all_labels += labels

            for sent in sents:
                if not self.is_eval:
                    tokenized = self.tokenizer(
                        sent,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=self.context_length,
                    )
                else:
                    tokenized = self.tokenizer(sent, return_tensors="pt")

                batch.append(tokenized)
                skill_idxs.append(skill)

        if not self.is_eval:
            batch, skill_idxs = self._shuffle_lists(batch, skill_idxs)
        return batch, skill_idxs, all_labels, output_idxs
    
    def _shuffle_lists(self, *ls):
        l = list(zip(*ls))
        random.shuffle(l)
        return zip(*l)

    def _generate_skill(self, digit, n):
        sents, labels = zip(*[self._generate_example(digit=digit) for _ in range(n)])
        if self.k == 4:
            output_idxs = [20] # NOTE 4 digit addn
        elif self.k == 3:
            output_idxs = [18] # NOTE 3 digit addn
        return sents, labels, output_idxs
    
    def _generate_example(self, digit):
        max_number = 10**self.k - 1
        a1_int = np.random.randint(0, max_number)
        a2_int = np.random.randint(0, max_number)
        A_int = a1_int + a2_int

        a1 = " ".join(self._get_digits(a1_int))
        a2 = " ".join(self._get_digits(a2_int))
        A = self._get_digits(A_int)[::-1]

        input_str = f"A = {a1} + {a2} , A {digit} = ?"
        output = int(A[digit])
        sent = f"Input: {input_str} Output: {output}"
        return sent, output

    def _get_digits(self, num):
        # convert numbers into a reversed list of digits, such that index 0 refers to the ones digit
        digits = str(num)
        if len(digits) < self.k:
            padding = "0" * self.k 
            digits = padding[:self.k-len(digits)] + digits
        return list(digits)



class AdditionTorchDataset(torch.utils.data.Dataset):
    def __init__(self, data, is_eval, skill_idxs=[], labels=None):
        self.data = data
        self.skill_idxs = skill_idxs
        self.is_eval = is_eval
        self.labels = labels
        
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.is_eval:
            # eval dataset
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
            # training dataset without skill idxs
            return {
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"],
            }
    def __len__(self):
        return len(self.data)
        
    