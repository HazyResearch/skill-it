import numpy as np
from tqdm import tqdm
import os
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Subset
import pickle
import pandas as pd
import wandb 
import numpy as np
from collections import OrderedDict

from .utils import get_train_dataset, get_tokenized_train_dataset, get_steps, create_optimizer_scheduler, get_train_dataloader, log_val_loss_per_skill



import sys
sys.path.append("..")

from .trainer import AbstractTrainer

from evaluator.ni_evaluator import NIMapDataset

class GroupCurriculumTrainer(AbstractTrainer):
    def train(
        self,
        args,
        logger,
        tokenizer,
        model,
        validation_data, 
        evaluator,
    ):
        """ Group curriculum learning baseline from https://arxiv.org/abs/2205.09898"""
        tokenized_val, output_idxs = validation_data.get_tokenized_dataset()
        train_data = get_train_dataset(args, logger, tokenizer)
        n_data = args.n_select if args.n_select != 0 else args.max_steps * args.batch_size
        tokenized_train = get_tokenized_train_dataset(args, train_data, n_data)
        
        ckpt_steps, total_steps = get_steps(args)            
                
        # we check if the list of losses and corresponding skill memberships are saved somewhere.
        curriculum_path = evaluator.curriculum_path
        curriculum_file = os.path.join(curriculum_path, f"seed_{args.selection_seed}_curriculum.pkl")
        skills_file = os.path.join(curriculum_path, f"seed_{args.selection_seed}_curriculum_skills.pkl")
        logger.info(f"Searching for curriculum in {curriculum_file}")
        if not os.path.exists(curriculum_file) or not os.path.exists(skills_file):
            logger.info(f"Curriculum not found, obtaining losses on training dataset now.")
            loss_list, skill_list, tokens = evaluator.evaluate(
                tokenized_train, None, None, output_idxs, train=True
            ) 
            if args.task_name == "ni":
                tokenized_train = tokens
        else:
            with open(curriculum_file, "rb") as f:
                logger.info("Curriculum file exists!")
                data = pickle.load(f)
                if args.task_name == "ni":
                    loss_list = data['losses']
                    tokenized_train = NIMapDataset(data['input_ids'])
                else:
                    loss_list = data
                    loss_list = np.array([loss.numpy() if torch.is_tensor(loss) else loss for loss in loss_list])
                    
            with open(skills_file, "rb") as f:
                logger.info("Skills file exists!")
                skill_list = pickle.load(f)
                
        loss_list = np.array(loss_list)
        assert not np.isnan(loss_list[0])
                    
        all_skills = sorted(np.unique(skill_list))
        loss_per_skill = OrderedDict()  
        for i, s in enumerate(all_skills):
            skill_idxs = np.where(skill_list == s)[0]
            avg_loss = loss_list[skill_idxs].mean()
            loss_per_skill[s]=avg_loss 
            
        # sort the group
        if args.curriculum:
            logger.info(f"Curriculum: Ordering training dataset per group from lowest to highest loss")
            loss_per_skill = OrderedDict(sorted(loss_per_skill.items(), key=lambda x: x[1]))
        else:
            logger.info(f"Anticurriculum: Ordering training dataset per group from highest to lowest loss")
            loss_per_skill = OrderedDict(sorted(loss_per_skill.items(), key=lambda x: x[1], reverse=True))
        
        ordered_skills = list(loss_per_skill.keys())
        logger.info(f"Ordered skills: {ordered_skills}")
        
        # The group curriculum paper uses many more training steps than regular training. For fair comparison we fix the number of training steps to be the same across our methods.
        n_skills = len(ordered_skills)
        total_unscaled_data = 0
        points_per_skill = OrderedDict()
        for i, s in enumerate(ordered_skills):
            skill_idxs = np.where(skill_list == s)[0]
            total_unscaled_data += len(skill_idxs) * (2 + (n_skills - (i + 1)) * args.mixing_frac)
            points_per_skill[s] = skill_idxs
        scaling_factor = total_steps * args.batch_size / total_unscaled_data
        
        current_skill = ordered_skills[0]
        ordered_skill_idxs = points_per_skill[current_skill]
        n_samples = int(scaling_factor * len(ordered_skill_idxs))
        
        logger.info(f"First ordered skill: {current_skill}. Number of samples: {n_samples}, number of steps: {n_samples / args.batch_size}")
        tokenized_train_init = Subset(
            tokenized_train, indices=np.random.choice(
                ordered_skill_idxs, size=n_samples, replace=False)
        )
    
        optimizer, lr_scheduler = create_optimizer_scheduler(model, args.lr, total_steps)
        
        train_dataloader = get_train_dataloader(args.task_name, tokenizer, tokenized_train_init, args.batch_size, args.slicer)   

        progress_bar = tqdm(range(total_steps))
        logging_steps = 10
        counter = 0
        max_grad_norm = 1.0
        model.zero_grad()
        for i in range(len(ordered_skills) + 1):        
            for idx, batch in enumerate(train_dataloader):
                model.train()
                batch = {k: v.cuda() for k, v in batch.items() if k not in ["skill_idxs"] and torch.is_tensor(v)}   
                outputs = model(**batch)
                loss = outputs.loss
                loss_all = torch.mean(loss)
                loss_all.backward()
            
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

                if counter % logging_steps == 0:
                    wandb.log({"train_loss": loss_all})
                if counter % ckpt_steps == 0:
                    loss_dict = evaluator.evaluate(
                        tokenized_val, counter, None, output_idxs
                    ) 
                    log_val_loss_per_skill(logger, loss_dict)                             
                    if args.task_name == "ni":
                        tokenized_val, _ = validation_data.get_tokenized_dataset()          
   
                counter += 1 
                progress_bar.update(1)
                
            if counter == total_steps:
                break 
            
            if i+1 == len(ordered_skills):
                # final round involves training over entire dataset randomly selected
                logger.info("Final training: training over the entire dataset")
                n_remaining = (total_steps - counter) * args.batch_size
                sampled_idxs = np.random.choice(np.arange(len(tokenized_train)), size=n_remaining, replace=False)
                tokenized_train_next = Subset(tokenized_train, indices = sampled_idxs)
                train_dataloader = get_train_dataloader(args.task_name, tokenizer, tokenized_train_next, args.batch_size, args.slicer)
            else:
                current_skill = ordered_skills[i+1]
                logger.info(f"Next skill is {current_skill}")
                ordered_skill_idxs = points_per_skill[current_skill]
                ordered_skill_idxs = np.random.choice(ordered_skill_idxs, size=int(scaling_factor*len(ordered_skill_idxs)), replace=False)
                # mix in fraction of previous groups 
                if args.mixing_frac != 0.0:
                    logger.info(f"Mixing in with fraction {args.mixing_frac}")
                    for j in range(i+1):
                        prev_skill = ordered_skills[j]
                        logger.info(f"Prev skill is {prev_skill}")
                        ordered_prev_skill_idxs = points_per_skill[prev_skill]
                        n_prev_skill = int(scaling_factor * args.mixing_frac * len(ordered_prev_skill_idxs)) 
                        ordered_prev_skill_idxs = np.random.choice(ordered_prev_skill_idxs, size=n_prev_skill, replace=False)
                        ordered_skill_idxs = np.concatenate([ordered_skill_idxs, ordered_prev_skill_idxs])
                
                ordered_skill_idxs = np.random.permutation(ordered_skill_idxs)
                tokenized_train_next = Subset(tokenized_train, indices=ordered_skill_idxs)
                train_dataloader =  get_train_dataloader(args.task_name, tokenizer, tokenized_train_next, args.batch_size, args.slicer)   

            
        loss_dict = evaluator.evaluate(
            tokenized_val, counter, None, output_idxs
        )    
        log_val_loss_per_skill(logger, loss_dict)         

