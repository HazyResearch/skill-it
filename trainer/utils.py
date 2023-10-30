from transformers import DataCollatorForLanguageModeling

from torch.utils.data import DataLoader, SequentialSampler, IterableDataset

from torch.optim import AdamW
from transformers.optimization import AdamW

from transformers import get_scheduler

from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import pandas as pd


import sys
sys.path.append("..")
from evaluator.utils import StringDataCollator

from dataset.addition_dataset import AdditionDataset 
from dataset.alpaca_dataset import AlpacaDataset
from dataset.lego_dataset import LegoDataset
from dataset.ni_dataset import NIDataset
from dataset.hf_dataset import HFDataset


def get_train_dataset(args, logger, tokenizer):
    if args.task_name == "ni":
        train_dataset = NIDataset(args, logger, tokenizer, args.selection_seed, args.sample_rule, False, args.train_data_dir)
    elif args.task_name == "lego":
        train_dataset = LegoDataset(args, logger, tokenizer, args.selection_seed, args.sample_rule, args.n_segment, False)
    elif args.task_name == "addition":
        train_dataset = AdditionDataset(args, logger, tokenizer, args.selection_seed, args.sample_rule, args.n_segment, False)
    elif args.task_name == "alpaca":
        train_dataset = AlpacaDataset(args, logger, tokenizer, args.selection_seed, args.sample_rule, False, args.train_data_dir)
    elif args.task_name == "law":
        train_dataset = HFDataset(args, logger, tokenizer, args.selection_seed, args.sample_rule, False, args.train_data_dir)
    return train_dataset
    
    
def get_tokenized_train_dataset(args, train_dataset, n_data):
    if args.task_name == "ni":
        tokenized_train = train_dataset.get_tokenized_dataset()
    elif args.task_name == "lego":
        tokenized_train = train_dataset.get_tokenized_dataset(n_data, include_skill_idxs=args.curriculum) 
    elif args.task_name == "addition":
        tokenized_train = train_dataset.get_tokenized_dataset(n_data, include_skill_idxs=args.curriculum)  
    elif args.task_name == "alpaca":
        tokenized_train = train_dataset.get_tokenized_dataset(n_data)
    elif args.task_name == "law":
        tokenized_train = train_dataset.get_tokenized_dataset(n_data)
    return tokenized_train
           
def get_steps(args):
    """Computes the number of steps per checkpoint and the total number of training steps."""
    ckpt_steps = (
        int((args.n_select * args.n_epochs / args.batch_size) / args.num_ckpts)
        if args.max_steps == -1
        else int(args.max_steps / args.num_ckpts)
    )
  
    total_steps = (
        args.max_steps
        if args.max_steps != -1
        else int(args.n_epochs * args.n_select / args.batch_size)
    )
    print(f"Total steps: {total_steps} Steps per checkpoint: {ckpt_steps}")
    
    if args.update_steps is not None:
        assert (args.update_steps % ckpt_steps == 0)
        assert (args.max_steps % args.update_steps == 0)    
    
    return ckpt_steps, total_steps

def get_update_steps(args, total_steps):
    """Computes the number of samples per update and the number of total updates (e.g., number of rounds T)."""
    update_size = args.update_steps * args.batch_size 
    n_updates = total_steps / args.update_steps
    return update_size, n_updates



def get_train_dataloader(task_name, tokenizer, tokenized_dataset, batch_size, slicer):
    """
        Returns DataLoader object for training data. 
    """  
    if task_name != "ni":
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    else:
        string_columns = ["task"] +  slicer
        data_collator = StringDataCollator(tokenizer, string_columns, mlm=False)

    train_sampler = SequentialSampler(tokenized_dataset)
    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        sampler=None if isinstance(tokenized_dataset, IterableDataset) else train_sampler,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    
def create_optimizer_scheduler(model, lr, max_steps):
    """
        Create AdamW optimizer and learning rate scheduler.
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=max_steps,
    )
    return optimizer, lr_scheduler

def aggregate_task_category(x):
    total_loss = x["loss"].apply(lambda x: sum(x)).sum()
    count_loss = x["loss"].apply(lambda x: len(x)).sum()
    metric_name = "task_loss"
    metric = total_loss/count_loss
    names = {metric_name: metric}    
    return pd.Series(names, index=[metric_name])


def log_val_loss_per_skill(logger, loss_dict):
    """ Logs the average loss per skill"""
    df= pd.DataFrame([{"task_idx": k, "loss": [values.numpy() for values in v]} for k, v in loss_dict.items()])
    df = df.groupby("task_idx").apply(lambda x: aggregate_task_category(x)).reset_index()
    df = df.sort_values(by="task_idx")                
    logger.info(df.head())
    
    

