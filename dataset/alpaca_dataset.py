import numpy as np
import pickle
import torch
import pandas as pd


from .utils import get_filter_skills, get_weights_from_graph

from .dataset import AbstractDataset
class AlpacaDataset(AbstractDataset):
    def __init__(
        self,
        args,
        logger,
        tokenizer,
        seed,
        sample_rule,
        is_eval,
        data_path,
    ):
        super().__init__(
            args, logger, tokenizer, seed, sample_rule, is_eval, data_path
        )
        self.logger.info(f"building Alpaca dataset.")

        self.k = args.k         
        
        with open(args.dev_split_path, "rb") as f:
            self.dev_split = pickle.load(f)

        with open(data_path, "rb") as f: 
            self.data = pickle.load(f)   
              
        self.set_skills(args)
        if not self.is_eval:
            self.set_proportions(args, args.proportions_file if args.proportions_file is not None else args.proportions)
            
    def set_skills(self, args):
        self.skills = sorted(self.data.skill.unique()) # always in alphabetical order 
        slices, _ = get_filter_skills(args.slice_list, args.exclude_slice, self.k)
        if slices is not None:
            if not self.is_eval or (self.is_eval and args.filter_val_skills):
                self.skills = np.array(slices) 
                self.data = self.data.loc[self.data.skill.isin(self.skills)]               
                
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
            # keep it as the true probs over the support 
            df = self.data.groupby("skill").size().reset_index()
            df[0] /= df[0].sum()
            self.proportions = df[0].values
        
    def get_tokenized_dataset(self, n_data=None):
        if self.is_eval:
            return self._get_tokenized_val()
        else:
            return self._get_tokenized_train(n_data)
        
    def _get_tokenized_val(self):
        eval_data = pd.DataFrame()
        for i, s in enumerate(self.skills):
            dev_samples = self.data.loc[self.dev_split[s]]
            eval_data = pd.concat([eval_data, dev_samples])
            
        tokenized =[{"skill": skill, "tokenized": self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=self.context_length, truncation=True)} for (skill, text) in eval_data[['skill', 'text']].values] 
        
        return AlpacaTorchDataset(tokenized, self.is_eval), None

    def _get_tokenized_train(self, n_data):        
        n_per_skill = (n_data * self.proportions).astype(int)
        n_per_skill[-1] = n_data - n_per_skill[:-1].sum()

        self.logger.info(f"Probabilities: {list(zip(self.skills, self.proportions))}")
        all_data = pd.DataFrame()
        for i, s in enumerate(self.skills):
            s_samples = self.data.loc[(self.data.skill == s) & (~self.data.index.isin(self.dev_split[s]))]
            if len(s_samples) < n_per_skill[i]:
                self.logger.warning(f"Not enough samples in slice {s}. size is {len(s_samples)}, requested is {n_per_skill[i]}")
            s_samples = s_samples.sample(n=min(len(s_samples), n_per_skill[i]))
            
            all_data = pd.concat([all_data, s_samples])
        
        all_data = all_data.sample(frac=1)
        tokenized =[{"skill": skill, "tokenized": self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=self.context_length, truncation=True)} for (skill, text) in all_data[['skill', 'text']].values] 
        return AlpacaTorchDataset(tokenized, self.is_eval)
        
class AlpacaTorchDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, is_eval):
        self.data = tokenized_data
        self.is_eval = is_eval
        
    def __getitem__(self, idx):
        data = self.data[idx]
        
        if self.is_eval:
            return {
                "input_ids": data['tokenized']['input_ids'],
                "attention_mask": data['tokenized']['attention_mask'],
                "skill": data['skill'],
            }
        else:
            return {
                "input_ids": data['tokenized']['input_ids'],
                "attention_mask": data['tokenized']['attention_mask']
            }
            
    def __len__(self):
        return len(self.data)
