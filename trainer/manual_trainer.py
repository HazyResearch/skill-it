import numpy as np

from tqdm import tqdm
import torch

import wandb


from torch.nn.utils import clip_grad_norm_
from .utils import get_train_dataset, get_tokenized_train_dataset, get_steps, create_optimizer_scheduler, get_train_dataloader, log_val_loss_per_skill
from .trainer import AbstractTrainer

class ManualTrainer(AbstractTrainer):
    def train(
        self,
        args,
        logger,
        tokenizer,
        model,
        validation_data,
        evaluator, 
    ):
        """ Training code that supports manual online data selection according to args.proportions_schedule."""
        tokenized_val, output_idxs = validation_data.get_tokenized_dataset()
        train_data = get_train_dataset(args, logger, tokenizer)
        ckpt_steps, total_steps = get_steps(args)
        optimizer, lr_scheduler = create_optimizer_scheduler(model, args.lr, total_steps)
        progress_bar = tqdm(range(total_steps))
        
        proportions_schedule = np.array(args.proportions_schedule).reshape((int(len(args.proportions_schedule) / args.k), args.k))
        assert(len(proportions_schedule) == int(args.max_steps / args.update_steps))
        # get first set of skills weights from args.proportions_schedule
        weights = proportions_schedule[0]
        train_data.set_proportions(args, weights)
        tokenized_train = get_tokenized_train_dataset(args, train_data, args.update_steps*args.batch_size)
        train_dataloader = get_train_dataloader(args.task_name, tokenizer, tokenized_train, args.batch_size, args.slicer)

        model.zero_grad()
        logging_steps = 10
        counter = 0
        max_grad_norm = 1.0
        segment_counter = 0
        logger.info(f"t: {counter}, proportions: {weights/sum(weights)}. ")
        while True:    
            dataloader_step = 0
            for idx, batch in enumerate(train_dataloader):
                model.train()
                batch = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
                outputs = model(**batch)
                loss = outputs.loss
                loss.mean().backward()

                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()
                
                if counter % logging_steps == 0:
                    wandb.log({"train_loss": loss})
                    
                if counter % ckpt_steps == 0:                    
                    loss_dict = evaluator.evaluate(
                        tokenized_val, counter, weights, output_idxs
                    )              
                    if args.task_name == "ni":
                        tokenized_val, _, _ = validation_data.get_tokenized_dataset()          
                    log_val_loss_per_skill(logger, loss_dict)         


                    
                dataloader_step += 1     
                counter += 1
                progress_bar.update(1)
                
                if dataloader_step == args.update_steps:
                    segment_counter += 1
                    break
            
            if counter == args.max_steps:
                break 
            
            # sample more training data according to next list of skills in args.proportions_schedule
            weights = proportions_schedule[segment_counter]
            logger.info(f"t: {counter}, proportions: {weights/sum(weights)}. ")
            
            train_data.set_proportions(args, weights)
            tokenized_train = get_tokenized_train_dataset(args, train_data, args.update_steps*args.batch_size)
            train_dataloader = get_train_dataloader(args.task_name, tokenizer, tokenized_train, args.batch_size, args.slicer)
            
        loss_dict = evaluator.evaluate(tokenized_val, counter, weights, output_idxs)    
        log_val_loss_per_skill(logger, loss_dict)         

