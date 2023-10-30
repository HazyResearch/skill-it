import numpy as np
from tqdm import tqdm
import os
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Subset
import pickle
import wandb 
import numpy as np
import pandas as pd

from .utils import get_train_dataset, get_tokenized_train_dataset, get_steps, create_optimizer_scheduler, get_train_dataloader, get_update_steps, log_val_loss_per_skill
from .trainer import AbstractTrainer
from evaluator.ni_evaluator import NIMapDataset

def _get_pacing(pacing_fn, pacing_a, pacing_b, update_size, n_updates, current_update):
    """Computes the number of top-k samples we can select from at a given update round. As training proceeds, this quantity will increase, but the rate at which this pool expands can be varied."""
    if pacing_fn == "linear":
        pool_size = update_size * current_update 
    elif pacing_fn == "log":
        n = n_updates * update_size
        pool_size = int(n * pacing_b + (1 - pacing_b) * n * (1 + 0.1 *( np.log(current_update/(pacing_a * n_updates) + np.e**(-10)))))
        if pool_size > n:
            pool_size = n
    return pool_size

class CurriculumTrainer(AbstractTrainer):
    def train(
        self,
        args,
        logger,
        tokenizer,
        model,
        validation_data,
        evaluator,
    ):
        """Curriculum learning baseline."""
        tokenized_val, output_idxs = validation_data.get_tokenized_dataset()
        train_data = get_train_dataset(args, logger, tokenizer)
        n_data = args.n_select if args.n_select != 0 else args.max_steps * args.batch_size
        tokenized_train = get_tokenized_train_dataset(args, train_data, n_data)
      
        # it is relatively time-consuming to iterate through every training sample and save their loss. We save the curriculum files so they can be used across curriculum, anticurriculum, and group curriculum baselines.
        curriculum_path = evaluator.curriculum_path
        curriculum_file = os.path.join(curriculum_path, f"seed_{args.selection_seed}_curriculum.pkl")
        logger.info(f"Searching for curriculum in {curriculum_file}")
        if not os.path.exists(curriculum_file):
            logger.info(f"Curriculum not found, obtaining losses on training dataset now.")
            loss_list, tokens = evaluator.evaluate(
                tokenized_train, None, None, output_idxs, train=True) 
            if args.task_name == "ni":
                tokenized_train = tokens # issues with NI IterableDataset
        else:
            with open(curriculum_file, "rb") as f:
                logger.info("Curriculum file exists!")
                data = pickle.load(f)
                if args.task_name == "ni":
                    loss_list = data['losses']
                    tokenized_train = NIMapDataset(data['input_ids']) # issues with NI IterableDataset
                else:
                    loss_list = data
                    loss_list = np.array([loss.numpy() if torch.is_tensor(loss) else loss for loss in loss_list])
        loss_list = np.array(loss_list)
        assert not np.isnan(loss_list[0])

        if args.curriculum:
            logger.info(f"Curriculum: Ordering training dataset from lowest to highest loss")
            ordered_idxs = loss_list.argsort()
        else:
            logger.info(f"Anticurriculum: Ordering training dataset from highest to lowest loss")
            ordered_idxs = np.flip(loss_list.argsort())
        
        # arrange dataset according to order 
        tokenized_train = Subset(tokenized_train, indices=ordered_idxs)
            
        ckpt_steps, total_steps = get_steps(args)
        update_size, n_updates = get_update_steps(args, total_steps)
        current_update = 1
        
        pool_size = _get_pacing(args.pacing_fn, args.pacing_a, args.pacing_b, update_size, n_updates, current_update)
        logger.info(f"Selecting {update_size} samples randomly from top {pool_size} scores")
        tokenized_pool = Subset(tokenized_train, indices=np.arange(pool_size))
        tokenized_train_init = Subset(tokenized_pool, indices=np.random.choice(np.arange(pool_size), size=update_size, replace=False))
        train_dataloader = get_train_dataloader(args.task_name, tokenizer, tokenized_train_init, args.batch_size, args.slicer)   
        
        optimizer, lr_scheduler = create_optimizer_scheduler(model, args.lr, total_steps)
        progress_bar = tqdm(range(total_steps))
        logging_steps = 10
        counter = 0
        max_grad_norm = 1.0
        model.zero_grad()
        while True:
            dataloader_step = 0
            for idx, batch in enumerate(train_dataloader):  
                model.train()
                batch = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v) and k != "skill_idxs"}
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
                              
                dataloader_step += 1
                counter += 1 
                progress_bar.update(1)
                if dataloader_step == args.update_steps:
                    break
            
            if counter == total_steps:
                break
            
            current_update += 1
            
            # update new dataset 
            pool_size = _get_pacing(args.pacing_fn, args.pacing_a, args.pacing_b, update_size, n_updates, current_update)
            logger.info(f"Selecting {update_size} samples randomly from top {pool_size} scores")                
            tokenized_pool = Subset(tokenized_train, indices=np.arange(pool_size))
            tokenized_train_next = Subset(tokenized_pool, indices = np.random.choice(np.arange(pool_size), size=update_size, replace=False) )
            train_dataloader =  get_train_dataloader(args.task_name, tokenizer, tokenized_train_next, args.batch_size, args.slicer)   
        
        loss_dict = evaluator.evaluate(
            tokenized_val, counter, None, output_idxs
        )    
        log_val_loss_per_skill(logger, loss_dict)         
