import numpy as np
from datasets import load_from_disk, concatenate_datasets

from .dataset import AbstractDataset

from .utils import get_filter_skills, get_weights_from_graph

class HFDataset(AbstractDataset):
    """ Constructs a subsampled HuggingFace dataset. Assumes that 'skill' is provided as a column name. """
    def __init__(
        self,
        args,
        logger,
        tokenizer,
        seed,
        sample_rule,
        is_eval,
        data_path
    ):
        super().__init__(
            args, logger, tokenizer, seed, sample_rule, is_eval, data_path
        )
        self.logger.info(f"building HuggingFace dataset.")

        self.k = args.k         
        self.data = load_from_disk(self.data_path) 
        self.set_skills(args)
        if not self.is_eval:
            self.set_proportions(args, args.proportions_file if args.proportions_file is not None else args.proportions)
        
    def set_skills(self, args):
        self.skills = sorted(self.data.unique('slice'))
        slices, _ = get_filter_skills(args.slice_list, args.exclude_slice, self.k)
        if slices is not None:
            if not self.is_eval or (self.is_eval and args.filter_val_skills):
                self.skills = np.array(slices) 
                skill_idxs = np.asarray([s in slices for s in self.data['slice']]).nonzero()[0]
                self.data = self.data.select(skill_idxs) # filter the data 
                
        self.k = len(self.skills)        
        self.logger.info(f"Remaining {self.k} skills:\n{self.skills}")

    
    def set_proportions(self, args, proportions):
        if self.sample_rule == "mixture":
            if ".npy" in proportions:
                proportions = np.load(proportions)
            elif (args.graph is not None or args.graph_path is not None) and not args.mw:
                proportions = get_weights_from_graph(args)
            elif proportions is None:
                raise ValueError("sample_rule is mixture, but neither proportion nor proportion file was provided.")     
            proportions = np.array(proportions)
            proportions /= sum(proportions)
            self.proportions = proportions
            assert len(self.proportions) == self.k
        elif self.sample_rule == "stratified":
            self.proportions = np.repeat(1.0 / self.k, self.k)
        else:            
            data_per_skill = []
            for i, s in enumerate(self.skills):
                data_per_skill.append(len(self.data.filter(
                    lambda x: x == s, input_columns="slice", num_proc=14
                )))
            data_per_skill = np.array(data_per_skill)
            self.proportions = data_per_skill / data_per_skill.sum()
            
    def get_tokenized_dataset(self, n_data=None):
        if self.is_eval:
            return self._get_tokenized_val()
        else:
            return self._get_tokenized_train(n_data)
        
    def _get_tokenized_val(self):
        return self.data.map(
            lambda x: self.tokenizer(
                x["text"],
                truncation=True,
                max_length=self.context_length,
                padding="max_length",
            ),
            batched=True,
        ), None

        
    def _get_tokenized_train(self, n_data):
        n_per_skill = (n_data * self.proportions).astype(int)
        n_per_skill[-1] = n_data - n_per_skill[:-1].sum()

        self.logger.info(f"Probabilities: {list(zip(self.skills, self.proportions))}")
        all_data = []
        for i, s in enumerate(self.skills):
            skill_data = self.data.filter(
                lambda x: x ==s, input_columns="slice", num_proc=10
            )
            if len(skill_data) < n_per_skill[i]:
                self.logger.warning(f"Not enough samples in slice {s}. size is {len(skill_data)}, requested is {n_per_skill[i]}")
                
            n = min(n_per_skill[i], len(skill_data))
            sample_idxs = np.random.choice(np.arange(len(skill_data)), size=n, replace=False) 
            all_data.append(skill_data.select(sample_idxs))
            
        self.data = concatenate_datasets(all_data).shuffle()
        return self.data.map(
            lambda x: self.tokenizer(
                x["text"],
                truncation=True,
                max_length=self.context_length,
                padding="max_length",
            ),
            batched=True,
            remove_columns=self.data.column_names,
        )
        
