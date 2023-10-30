
import os
from collections import defaultdict
from torch.utils.data import DataLoader, SequentialSampler, Dataset
import torch
import numpy as np
from tqdm import tqdm
import pickle
import string 

from .evaluator import AbstractEvaluator
from .utils import StringDataCollator, save_loss, save_weights, save_curriculum, save_skill_list, set_curriculum_str


class NIEvaluator(AbstractEvaluator):
    def _set_results_path(self):
        method_name = self._set_header_str()            
        method_name = set_curriculum_str(self.args, method_name)
        method_name = self._set_method_str(method_name)

        self.result_path = os.path.join(self.output_dir_path, method_name)
        self.logger.info(f"Output path is {self.result_path}")
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            
    def _set_curriculum_path(self):
        method_name = self._set_header_str()
        if self.args.group_curriculum or self.args.curriculum or self.args.anticurriculum:
            method_name += "curriculum_"
        method_name = self._set_method_str(method_name)

        self.curriculum_path = os.path.join(self.output_dir_path, method_name)
        self.logger.info(f"Curriculum path is {self.curriculum_path}")
        
        if not os.path.exists(self.curriculum_path):
            os.makedirs(self.curriculum_path)    
  
    def _set_header_str(self):
        method_name = f"{self.args.task_name}_"
        if self.args.xlingual:
            method_name += "xlingual_"
        elif self.args.ni_test:
            method_name += "ood_"
        elif len(self.args.slicer) > 1:
            slicer_str = "_".join(self.args.slicer)
            method_name += f"{slicer_str}_"
        return method_name
  
    def _set_method_str(self, method_name):    
        if self.args.model_name != "EleutherAI/gpt-neo-125M":
            method_name += self.args.model_name.split("/")[-1] + "_"
        
        if self.args.n_select != 0:
            method_name += f"{self.args.n_select}_" 
        else:
            method_name += f"{self.args.max_steps}_"

        if self.args.sample_rule is not None:
            method_name += f"{self.args.sample_rule}_"
        
        # we have a list of slices 
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
        
        if self.args.proportions is not None and self.args.graph is None:
            proportions_str = "".join([str(int(i)) for i in self.args.proportions])
            method_name += f"weights_{proportions_str}_"

        if self.args.proportions_schedule is not None:
            proportions_str = "".join([str(int(i)) for i in self.args.proportions_schedule])
            method_name += f"_weightschedule_{proportions_str}"
            
        if self.args.target_mask is not None:
            target_str = "".join([str(int(i)) for i in self.args.target_mask])
            method_name += f"targetmask_{target_str}_mean_"
            
        if self.args.graph is not None:
            method_name += "graph_"
            graph_str = "".join([str(g) for g in self.args.graph])
            method_name += f"_{graph_str}"
        if self.args.graph_path is not None:
            graph_path_str = ".".join(self.args.graph_path.split("/")[-1].split(".")[:-1])
            method_name += f"{graph_path_str}_"
            
            
        if self.args.proportions_file is not None:
            prop_str = ".".join(self.args.proportions_file.split(".")[:-1])
            method_name += f"_{prop_str}"

        
        if self.args.mw:
            method_name += f"greedy_{self.args.update_steps}_"
            if self.args.eta_schedule:
                method_name += f"eta_schedule_{self.args.eta}_"
            else:
                method_name += f"eta_{self.args.eta}_" 
            method_name += f"lookback_{self.args.mw_window}_checkpoints_"
            if self.args.mw_init is not None:
                mw_init_str = "".join([str(int(i)) for i in self.args.mw_init])
                method_name += f"mwinit_{mw_init_str}_"
            if self.args.ignore_lone_nodes:
                method_name += "ignorelonenodes_"      
            if self.args.normalize_loss:
                method_name += "normalizeloss_" 
            if self.args.dynamic_lambda:
                method_name += "dynamiclambda_"
        else:
            method_name += "static_"
            if self.args.eta is not None:
                method_name += f"eta_{self.args.eta}_"

        
        if self.args.lr != 5e-5:
            method_name += f"lr_{self.args.lr}_"
            
        if self.args.one_sample_per_window:
            method_name += "onesamplerperwindow_"
            
        if method_name.endswith("_"):
            method_name = method_name[:-1]

        return method_name     


    def evaluate(self, tokenized_data, counter, weights, output_idxs=None, train=False):
        tokenized_data.reset_seed() 
        if train:
            return self._evaluate_train(tokenized_data)
        else:
            return self._evaluate_val(tokenized_data, counter, weights)
        
        
    def _evaluate_train(self, tokenized_data):
        self.model.eval()
        loss_list = []
        input_ids_list = []
        skill_list = []
        
        train_dataloader = self._make_dataloader(tokenized_data)        
        for i, data in tqdm(enumerate(train_dataloader)):
            if i == self.args.max_steps:
                break
            
            if len(self.args.slicer) == 1:
                skills = data[self.args.slicer[0]]
            else:
                skills = zip(*[data[slicer_dim] for slicer_dim in self.args.slicer])
            input_ids = data['input_ids'].to("cuda")
            labels = data['input_ids'].clone().to("cuda")
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels)
                losses = outputs.loss.cpu()
                if i == 0:
                    context_length = int(len(losses) / self.args.batch_size)
                    
                losses = losses.view(-1, context_length)
                keep = losses != 0
                losses = (losses * keep).sum(dim = 1)/keep.sum(dim=1)
                
            loss_list.append(losses.numpy())
            skill_list.extend(skills)
            inputs = list(data['input_ids'])
            input_ids_list.extend(inputs)
            
        loss_list = np.array(loss_list).flatten()
        skill_list = np.array(skill_list).flatten()
            
        ni_data = NIMapDataset(input_ids_list)
        
        loss_dict = {"losses": loss_list, "input_ids": input_ids_list}
        
        loss_list_path = save_curriculum(loss_dict, self.curriculum_path, self.args.selection_seed)
        self.logger.info(f"Saving curriculum losses list to {loss_list_path}")
        
        skill_list_path = save_skill_list(skill_list, self.curriculum_path, self.args.selection_seed)
        self.logger.info(f"Saving curriculum skills list to {skill_list_path}")   
                 
        if self.args.group_curriculum:
            return loss_list, skill_list, ni_data
        else:
            return loss_list, ni_data
        
        
    def _evaluate_val(self, tokenized_data, counter, weights):
        self.model.eval()
        
        loss_dict = defaultdict(list)
        val_dataloader = self._make_dataloader(tokenized_data)
        
        for i, data in tqdm(enumerate(val_dataloader)):
            if len(self.args.slicer) == 1:
                skills = data[self.args.slicer[0]]
            else:
                skills = zip(*[data[slicer_dim] for slicer_dim in self.args.slicer])
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
            
            for i, skill in enumerate(skills):
                if isinstance(skill, tuple):
                    key_str = "_".join(skill)
                else:
                    key_str = skill
                loss_dict[key_str].append(losses[i])
                
                
        loss_path = save_loss(loss_dict, self.result_path, self.args.selection_seed, counter)
        save_weights(weights, self.result_path, self.args.selection_seed, counter)
        return loss_dict

    def _make_dataloader(self, tokenized_data):
        string_columns = ["task"] + self.args.slicer
        data_collator = StringDataCollator(self.tokenizer, string_columns, mlm=False)
        dataloader = DataLoader(
            tokenized_data,
            batch_size = self.args.batch_size,
            sampler = None,
            collate_fn = data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True
        )                  
        return dataloader               

        

class NIMapDataset(Dataset):
    def __init__(self, input_ids_list):
        self.input_ids_list = input_ids_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx]
        }