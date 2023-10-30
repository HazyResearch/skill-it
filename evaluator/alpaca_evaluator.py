
import os
from collections import defaultdict
from torch.utils.data import DataLoader, SequentialSampler
import torch
from tqdm import tqdm

from .evaluator import AbstractEvaluator
from .utils import StringDataCollator, save_loss, save_weights


class AlpacaEvaluator(AbstractEvaluator):
    def _set_results_path(self):
        method_name = f"{self.args.task_name}_"
        
        if self.args.n_select != 0:
            method_name += f"{self.args.n_select}_"
        else:
            method_name += f"{self.args.max_steps}_"
            
        if self.args.sample_rule is not None:
            method_name += f"{self.args.sample_rule}_"
        
        if self.args.slice_list is not None:
            if len(self.args.slice_list) == 1:
                slice_list = self.args.slice_list[0]
                if "txt" in slice_list:
                    method_name += f"{slice_list.split('/')[-1]}_"
                else:
                    method_name += f"{slice_list}_"
            else:
                slice_list = "_".join(self.args.slice_list)
                if len(slice_list) > 50:
                    slice_list = slice_list[:50] # truncate 
                method_name += f"{slice_list}_"
        
        if self.args.proportions is not None:
            proportions_str = "".join([str(int(i)) for i in self.args.proportions])
            method_name += f"weights_{proportions_str}_"
                    
        if self.args.graph is not None:
            method_name += "graph_"
        if self.args.graph_path is not None:
            graph_path_str = self.args.graph_path.split("/")[-1].split(".")[0]
            method_name += f"{graph_path_str}_" 
            
        if self.args.mw:
            method_name += f"greedy_{self.args.update_steps}_"
            if self.args.eta_schedule:
                method_name += f"eta_schedule_{self.args.eta}_"
            else:
                method_name += f"eta_{self.args.eta}_"
            method_name += f"lookback_{self.args.mw_window}_checkpoints_"
            if self.args.normalize_loss:
                method_name += "normalizeloss_"
            if self.args.dynamic_lambda:
                method_name += "dynamiclambda_"
            if self.args.ignore_lone_nodes:
                method_name += "ignorelonenodes_"
        else:
            method_name += "_static"
            if self.args.eta is not None:
                method_name += f"eta_{self.args.eta}_"
            
        if self.args.lr != 5e-5:
            method_name += f"lr_{self.args.lr}_"
      
        if method_name.endswith("_"):
            method_name = method_name[:-1]
        
        self.result_path = os.path.join(self.output_dir_path, method_name)
        self.logger.info(f"Output path is {self.result_path}")
        
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)    
                
    
    def evaluate(self, tokenized_data, counter, weights, output_idxs=None, train=False):
        return self._evaluate_val(tokenized_data, counter, weights)
   
    
    def _evaluate_val(self, tokenized_data, counter, weights):
        self.model.eval()
        loss_dict = defaultdict(list)
        
        val_dataloader = self._make_dataloader(tokenized_data)
        for i, data in tqdm(enumerate(val_dataloader)):
            skills = data['skill']
            input_ids = data['input_ids'].to('cuda')
            labels = data['input_ids'].clone().to('cuda')
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels)
                losses = outputs.loss.cpu()
                if i == 0:
                    context_length = int(len(losses) / self.args.batch_size)
                    
                losses = losses.view(-1, context_length)
                keep = losses != 0
                losses = (losses * keep).sum(dim = 1) / keep.sum(dim = 1)
                
            for j, skill in enumerate(skills):
                loss_dict[skill].append(losses[j])   
                
                
        loss_path = save_loss(loss_dict, self.result_path, self.args.selection_seed, counter)
        save_weights(weights, self.result_path, self.args.selection_seed, counter)
        return loss_dict
    
    def _make_dataloader(self, tokenized_data):
        string_columns = ["skill"]
        data_collator = StringDataCollator(self.tokenizer, string_columns, mlm=False)
        sampler = SequentialSampler(tokenized_data)
        dataloader = DataLoader(
            tokenized_data,
            batch_size=self.args.batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True
        )
        return dataloader