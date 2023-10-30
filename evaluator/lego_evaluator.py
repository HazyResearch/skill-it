
import os
from collections import defaultdict
from torch.utils.data import DataLoader, SequentialSampler
import torch
import numpy as np
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
import string 

from .evaluator import AbstractEvaluator

from .utils import save_loss, save_weights, save_embeddings, save_labels, save_predictions, save_curriculum, save_skill_list, set_curriculum_str


class LegoEvaluator(AbstractEvaluator):
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
        return f"{self.args.task_name}_{self.args.k}_"


    def _set_method_str(self, method_name):
        if self.args.model_name != "EleutherAI/gpt-neo-125M":
            method_name += self.args.model_name.split("/")[-1] + "_"
        
        if self.args.n_select != 0:
            method_name += f"{self.args.n_select}_" 
        else:
            method_name += f"{self.args.max_steps}_"
                
        if self.args.sample_rule is not None:
            method_name += f"{self.args.sample_rule}_"
                        
        if self.args.n_segment > 1:
            method_name += f"n_segment_{self.args.n_segment}_"
        
        if self.args.batch_size != 32:
            method_name +=f"{self.args.batch_size}_"
            
        if self.args.target_mask is not None:
            target_str = "".join([str(int(i)) for i in self.args.target_mask])
            method_name += f"targetmask_{target_str}_mean_"
        
        if self.args.lego_graph is not None:
            method_name += f"{self.args.lego_graph}_"
                
        if self.args.proportions is not None and self.args.graph is None:
            proportions_str = "".join([str(int(i)) for i in self.args.proportions])
            method_name += f"weights_{proportions_str}_"
 
        if self.args.proportions_schedule is not None:
            proportions_str = "".join([str(int(i)) for i in self.args.proportions_schedule])
            method_name += f"_weightschedule_{proportions_str}"
           
        if self.args.segment_proportions is not None:
            proportions_str = "".join([str(int(i)) for i in self.args.segment_proportions])
            method_name += f"segments_{proportions_str}_"
            
        if self.args.graph is not None:
            method_name += "graph_"
            graph_str = "".join([str(int(i)) for i in self.args.graph])
            method_name += f"{graph_str}_"
        if self.args.graph_path is not None:
            graph_path_str = ".".join(self.args.graph_path.split("/")[-1].split(".")[:-1])
            method_name += f"{graph_path_str}_"
            
        if self.args.mw:
            method_name += f"greedy_{self.args.update_steps}_"
            if self.args.eta_schedule:
                method_name += f"eta_schedule_{self.args.eta}_"
            else:
                method_name += f"eta_{self.args.eta}_" 
            method_name += f"lookback_{self.args.mw_window}_"
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
            method_name += f"lr_{self.args.lr}"
            
        if method_name.endswith("_"):
            method_name = method_name[:-1]
            
        return method_name
                            
        
    def evaluate(self, tokenized_data, counter, weights, output_idxs=None, train=False):
        if train:
            return self._evaluate_train(tokenized_data, output_idxs)
        else:
            loss_dict = self._evaluate_val(tokenized_data, output_idxs, counter, weights)
            return loss_dict 
    
    def _evaluate_train(self, tokenized_data, output_idxs):
        self.model.eval()
        skill_list = []
        loss_list = []
        train_dataloader = self._make_dataloader(tokenized_data)
        for i, data in tqdm(enumerate(train_dataloader)):
            input_ids = data['input_ids'].to("cuda")
            labels = data['input_ids'].clone().to('cuda')
            mask = list(set(np.arange(len(labels[0, 0, :]))).difference(set(output_idxs)))
            labels[:, :, mask] = -100
            skill_idx = torch.flatten(data['skill_idxs']).numpy()
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels, output_hidden_states=True, return_dict=True)
                losses = outputs.loss.cpu()
                if i == 0:
                    context_length = int(len(losses)/self.args.batch_size)
                    
                losses = losses.view(-1, context_length)
                keep = losses != 0
                losses = (losses * keep).sum(dim = 1)/keep.sum(dim = 1)
                
            loss_list.extend(losses.numpy())
            skill_list.append(skill_idx)
            
        skill_list = np.array(skill_list).flatten()   
        
        loss_list_path = save_curriculum(loss_list, self.curriculum_path, self.args.selection_seed)
        self.logger.info(f"Saving curriculum losses list to {loss_list_path}")
        skill_list_path = save_skill_list(skill_list, self.curriculum_path, self.args.selection_seed)
        self.logger.info(f"Saving curriculum skills list to {skill_list_path}")
            
        if self.args.group_curriculum:
            return loss_list, skill_list, None 
        else:
            return loss_list, None
            
        
    def _evaluate_val(self, tokenized_data, output_idxs, counter, weights):
        self.model.eval()
        
        loss_dict = defaultdict(list)
        emb_dict = defaultdict(list) 
        prediction_dict = defaultdict(list)
        labels_dict = defaultdict(list)
        
        val_dataloader = self._make_dataloader(tokenized_data)
        for i, data in tqdm(enumerate(val_dataloader)):
            input_ids = data['input_ids'].to("cuda")
            pred_labels = data['output_labels']
            labels = data['input_ids'].clone().to("cuda")
            mask = list(set(np.arange(len(labels[0, 0, :]))).difference(set(output_idxs)))
            labels[:, :, mask] = -100
            skill_idx = torch.flatten(data['skill_idxs']).numpy()
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels, output_hidden_states=True, return_dict=True)
                losses = outputs.loss.cpu()
                if i == 0:
                    context_length = int(len(losses)/ self.args.batch_size)
                losses = losses.view(-1, context_length)
                keep = losses != 0 
                losses = (losses * keep).sum(dim=1)/keep.sum(dim=1)
                if self.args.get_embs:
                    embs = outputs['hidden_states'][-1][:, :, -1, :].cpu().numpy().reshape((-1, 768))
                    
            encodings = {"input_ids": input_ids, "attention_mask": data['attention_mask'].to("cuda")}
            predictions = self._query(0.0, 3, output_idxs, encodings)
              
                    
            for j, t in enumerate(skill_idx):
                if self.args.get_embs:
                    emb_dict[t].append(embs[j])
                    
                loss_dict[t].append(losses[j])
                
                prediction_dict[t].append(predictions[j])
                labels_dict[t].append(pred_labels[j])
                  
        loss_path = save_loss(loss_dict, self.result_path, self.args.selection_seed, counter)
        save_weights(weights, self.result_path, self.args.selection_seed, counter)
        if self.args.get_embs:
            save_embeddings(emb_dict, self.result_path, self.args.selection_seed, counter)
        save_predictions(prediction_dict, self.result_path, self.args.selection_seed, counter)
        save_labels(labels_dict, self.result_path)
                   
        return loss_dict
    
    def _query(self, temperature, max_tokens, output_idxs, encodings):
        do_sample = True 
        if temperature == 0:
            do_sample = False 
            
        input_dims = encodings['input_ids'].shape
        encodings['input_ids'] = encodings['input_ids'].view(input_dims[0], -1)[:, :output_idxs[0]]
        encodings['attention_mask'] = encodings['attention_mask'].view(input_dims[0], -1)[:, :output_idxs[0]]
        with torch.no_grad():
            gen_tokens = self.model.generate(
                **encodings,
                do_sample=do_sample,
                temperature=temperature,
                max_length=len(encodings['input_ids'][0]) + max_tokens,
                pad_token_id = self.tokenizer.eos_token_id) 
        gen_texts = self.tokenizer.batch_decode(gen_tokens)
        predictions = [gen_text.split("= val ")[-1].strip().strip(string.punctuation)[0] for gen_text in gen_texts]
        predictions = [int(prediction) if prediction.isnumeric() else -1 for prediction in predictions]
        return predictions
                
        
    def _make_dataloader(self, tokenized_data):
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        sampler = SequentialSampler(tokenized_data)
        dataloader = DataLoader(
            tokenized_data,
            batch_size = self.args.batch_size,
            sampler = sampler,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        return dataloader