import numpy as np
from tqdm import tqdm
import torch
import wandb
import torch
from torch.nn.utils import clip_grad_norm_
import pandas as pd 

from .utils import get_steps, create_optimizer_scheduler, aggregate_task_category, get_train_dataloader, get_train_dataset, get_tokenized_train_dataset, log_val_loss_per_skill

from .trainer import AbstractTrainer

class MWTrainer(AbstractTrainer):
    def train(
        self,
        args,
        logger,
        tokenizer,
        model,
        validation_data,
        evaluator,
    ):
        """ Skill-It online data selection, plus algorithmic variations."""    
        tokenized_val, output_idxs = validation_data.get_tokenized_dataset()
        train_data = get_train_dataset(args, logger, tokenizer)
  
        ckpt_steps, total_steps = get_steps(args)
        optimizer, lr_scheduler = create_optimizer_scheduler(model, args.lr, total_steps)
            
        all_losses = []
        if args.initialize_loss:
            all_losses.append(np.ones(args.k))
        
        if args.graph is not None:
            side_length = int(np.sqrt(len(args.graph)))
            graph = np.array(args.graph).reshape((side_length, side_length)).astype(float)
            for i in range(side_length):
                for j in range(side_length):
                    if i != j and graph[i, j] == 1:
                        graph[i, j] = 0.5 # For synthetics, we make off-diagonal nonzero entries have weight 0.5
            if args.ignore_lone_nodes:
                lone_node_idxs = np.array([i for i in range(args.k) if graph[i, i] == 1 and graph[i, :].sum() == 1 and graph[:, i].sum() == 1 ])
                mw_idxs = np.setdiff1d(np.arange(args.k), lone_node_idxs)
                graph = graph[mw_idxs][:, mw_idxs]
        elif args.graph_path is not None:
            graph = np.load(args.graph_path).astype(float)
            n, m = graph.shape
            for i in range(n):
                for j in range(m):
                    if i != j and graph[i, j] == 1:
                        graph[i, j] = 0.5
                        
            if args.ignore_lone_nodes:
                lone_node_idxs = np.array([i for i in range(args.k) if graph[i, i] == 1 and graph[i, :].sum() == 1 and graph[:, i].sum() == 1 ])
                mw_idxs = np.setdiff1d(np.arange(args.k), lone_node_idxs)
                graph = graph[mw_idxs][:, mw_idxs]
        else:
            graph = np.eye(args.k)
        
        
        if args.target_mask is not None:
            args.target_mask = np.array([int(i) for i in args.target_mask])
            target_idxs = np.where(args.target_mask == 1)[0]
            graph = graph[:, target_idxs]
            logger.info(f"Target mask is set to {args.target_mask}")
        
        logger.info(f"Using dependency graph:\n{graph}")

            
        if args.mw_prior is not None:
            weights_init = np.array(args.mw_prior)
        else:
            weights_init = np.ones(graph.shape[0])
            logger.info(f"weights init are {weights_init}")
    
    
        if args.ni_test:
            loss_init = np.ones(graph.shape[1])
        elif args.target_mask is not None:
            loss_init = np.ones(len(target_idxs))
        else:
            loss_init = weights_init
    
        if args.mw_init is not None:
            mw_init = np.array(args.mw_init)
            mw_init /= sum(mw_init)
            # weights = np.multiply(weights_init, np.exp(eta * mw_init))
            weights = mw_init
            # all_losses.append(mw_init)
        else:
            if args.ignore_lone_nodes:
                # lone nodes get the same amount of weight as with uniform, 1/args.k
                weights_lone = np.repeat(1/args.k, len(lone_node_idxs))

                # do MW update on mw_idxs
                weights_mw = np.multiply(weights_init[mw_idxs], np.exp(args.eta * graph.dot(weights_init[mw_idxs])))
                # only allocate 1 - len(lone_nodes)/args.k mass for MW nodes 
                weights_mw_total_mass = 1 - weights_lone.sum()
                weights_mw /= sum(weights_mw)
                weights_mw *= weights_mw_total_mass
                
                weights = weights_init 
                weights[mw_idxs] = weights_mw
                weights[lone_node_idxs] = weights_lone
                assert weights.sum() == 1
                logger.info(f"{len(mw_idxs)} nodes are connected, {len(lone_node_idxs)} nodes are not")
                logger.info(f"weights_mw: {weights_mw}")

            else:
                weights = np.multiply(weights_init, np.exp(args.eta * graph.dot(loss_init)))
                logger.info(f"Loss init is {loss_init}, weights are {weights}")

        train_data.set_proportions(args, weights)
        tokenized_train = get_tokenized_train_dataset(args, train_data, args.update_steps*args.batch_size)
        train_dataloader = get_train_dataloader(args.task_name, tokenizer, tokenized_train, args.batch_size, args.slicer)
        
        model.zero_grad()
        logging_steps = 10
        counter = 0
        max_grad_norm = 1.0
        progress_bar = tqdm(range(total_steps))
        logger.info(f"t={counter}, new data distribution={weights/sum(weights)}. ")
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
                        tokenized_val, _ = validation_data.get_tokenized_dataset()          
                    
                    # compute losses every ckpt_steps 
                    df= pd.DataFrame([{"task_idx": k, "loss": [values.numpy() for values in v]} for k, v in loss_dict.items()])
                    df = df.groupby("task_idx").apply(lambda x: aggregate_task_category(x)).reset_index()
                    df = df.sort_values(by="task_idx")
                    if args.target_mask is not None:
                        df = df.loc[df.index.isin(target_idxs)] # filter to only be the losses we care about 
                    logger.info(df.head())
                    
                    if counter == 0:
                        loss_0 = df.task_loss.values
                        if args.initialize_loss:
                            all_losses[0] = all_losses[0] * loss_0.mean()
                        if args.ignore_lone_nodes:
                            loss_0 = np.array(loss_0)[mw_idxs]
                    elif counter == ckpt_steps:
                        loss_0 = df.task_loss.values
                        if args.ignore_lone_nodes:
                            loss_0 = np.array(loss_0)[mw_idxs]
     
                    all_losses.append(df.task_loss.values)
                    
                dataloader_step += 1     
                counter += 1
                progress_bar.update(1)
                        
                if dataloader_step == args.update_steps:
                    break
            
            if counter == total_steps:
                break 
            
            # update skills mixture 
            idx = len(all_losses)            
            eta_t = args.eta / (all_losses[-1].sum() / loss_0.sum()) if args.eta_schedule else args.eta
            if args.mw_window >= 0:
                if args.ignore_lone_nodes:                
                    if args.normalize_loss:
                        weights_mw = np.multiply(weights_init[mw_idxs], np.exp(eta_t * graph.dot(np.divide(np.array(all_losses[max(0, idx - args.mw_window): idx])[:, mw_idxs].sum(axis=0), loss_0))))
                    else:
                        weights_mw = np.multiply(weights_init[mw_idxs], np.exp(eta_t * graph.dot(np.array(all_losses[max(0, idx - args.mw_window): idx])[:, mw_idxs].sum(axis=0))))
                    weights_mw /= sum(weights_mw)
                    weights_mw *= weights_mw_total_mass
                    weights[mw_idxs] = weights_mw
                else:
                    if args.target_mask is not None:
                        loss_arr = np.array(all_losses[max(0, idx - args.mw_window) : idx]).mean(axis=0)
                    else:
                        loss_arr = np.array(all_losses[max(0, idx - args.mw_window): idx]).sum(axis=0)
                
                    if args.normalize_loss:
                        weights = np.multiply(weights_init, np.exp(eta_t * graph.dot(np.divide(loss_arr, loss_0))))
                    else:
                        weights = np.multiply(weights_init, np.exp(eta_t * graph.dot(loss_arr)))

            else:
                raise NotImplementedError("Standard multiplicative weights not supported")
            if args.dynamic_lambda:
                n, m = graph.shape
                for i in range(n):
                    for j in range(m):
                        if args.target_mask is not None and (j, i) in enumerate(target_idxs):
                            continue # do not change weight from target skill to target skill
                        if graph[i, j] > 0:
                            if (n == m or (n == 1 or m == 1)) and i != j:
                                graph[i, j] /= 2
                            elif n != m:
                                graph[i, j] /= 2             
                logger.info(f"Dampening off-diagonal entries. Updated graph: {graph}")

            logger.info(f"t={counter}, new data distribution={weights/sum(weights)}")
            
            # create new dataset for next round
            train_data.set_proportions(args, weights)
            tokenized_train = get_tokenized_train_dataset(args, train_data, args.update_steps*args.batch_size)
            train_dataloader = get_train_dataloader(args.task_name, tokenizer, tokenized_train, args.batch_size, args.slicer)
                        
        loss_dict = evaluator.evaluate(
            tokenized_val, counter, weights, output_idxs
        )      
        log_val_loss_per_skill(logger, loss_dict)         

            