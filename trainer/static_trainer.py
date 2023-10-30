from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import wandb 
import pandas as pd

from .utils import get_train_dataset, get_tokenized_train_dataset, get_steps, create_optimizer_scheduler, get_train_dataloader, log_val_loss_per_skill
from .trainer import AbstractTrainer 

class StaticTrainer(AbstractTrainer):
    def train(
        self,
        args,
        logger,
        tokenizer,
        model,
        validation_data,
        evaluator, 
    ):
        """Standard Pytorch training and evaluation code without any online sampling."""
        tokenized_val, output_idxs = validation_data.get_tokenized_dataset()
        train_data = get_train_dataset(args, logger, tokenizer)
        n_data = args.n_select if args.n_select != 0 else args.max_steps * args.batch_size
        tokenized_train = get_tokenized_train_dataset(args, train_data, n_data)
        
        train_dataloader = get_train_dataloader(args.task_name, tokenizer, tokenized_train, args.batch_size, args.slicer)   
        ckpt_steps, total_steps = get_steps(args)
        optimizer, lr_scheduler = create_optimizer_scheduler(model, args.lr, total_steps)
        
        progress_bar = tqdm(range(total_steps))
        logging_steps = 10
        counter = 0
        max_grad_norm = 1.0
        model.zero_grad()
        for i, batch in enumerate(train_dataloader):
            model.train()
            batch = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}   
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
                loss_dict = evaluator.evaluate(tokenized_val, counter, None, output_idxs)   
                if args.task_name == "ni":
                    # we need to create a new validation dataset for Natural Instructions since it is an IterableDataset
                    tokenized_val, _ = validation_data.get_tokenized_dataset()    
                log_val_loss_per_skill(logger, loss_dict)    
            counter += 1
            progress_bar.update(1)

            if counter == total_steps:
                break
    
        loss_dict = evaluator.evaluate(tokenized_val, counter, None, output_idxs) 
        log_val_loss_per_skill(logger, loss_dict) 