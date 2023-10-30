"""Utility functions."""
import os
import logging

from typing import Optional, Tuple, Union

import torch 

from transformers import GPTNeoForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast

from torch.nn import CrossEntropyLoss

from trainer.curriculum_trainer import CurriculumTrainer
from trainer.group_curriculum_trainer import GroupCurriculumTrainer
from trainer.manual_trainer import ManualTrainer 
from trainer.mw_trainer import MWTrainer 
from trainer.static_trainer import StaticTrainer

from dataset.addition_dataset import AdditionDataset
from dataset.alpaca_dataset import AlpacaDataset
from dataset.lego_dataset import LegoDataset
from dataset.ni_dataset import NIDataset
from dataset.hf_dataset import HFDataset

from evaluator.evaluator2class import Evaluator2Class

def make_output_dir(output_dir, session_id, run_id):
    dir_path = os.path.join(output_dir, session_id if session_id is not None else run_id)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def get_logger(dir_path):
    # Create a logger
    logger = logging.getLogger("LLM-based evaluation")
    logger.setLevel(logging.INFO)

    # Create a file handler that writes to output.log
    file_handler = logging.FileHandler(os.path.join(dir_path, "output.log"))
    file_handler.setLevel(logging.INFO)

    # Create a stream handler that prints to the screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create a formatter for the log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    logger.propagate = False

    return logger

def get_trainer(args):
    if args.mw:
        trainer = MWTrainer()
    elif args.proportions_schedule is not None:
        trainer = ManualTrainer()
    elif args.group_curriculum:
        trainer = GroupCurriculumTrainer()
    elif not args.group_curriculum and (args.curriculum or args.anticurriculum):
        trainer = CurriculumTrainer()
    else:
        trainer = StaticTrainer()
    return trainer 

def get_val_dataset(args, logger, tokenizer):
    if args.task_name == "ni":
        seed = args.selection_seed
        val_data = NIDataset(
            args, logger, tokenizer,
            seed, sample_rule="random", is_eval=True, data_path=args.val_data_dir,
            dev_split_path="./aux_data/test_dev_split_map.pkl" if args.ni_test else None,
            ni_task_info_path="./aux_data/ni_default_task_info_test.pkl" if args.ni_test else None
        )
    elif args.task_name == "lego":
        seed = 42
        val_data = LegoDataset(
            args, logger, tokenizer, seed, sample_rule="stratified", n_segment=1, is_eval=True,
        )
    elif args.task_name == "addition":
        seed = 420
        val_data = AdditionDataset(args, logger, tokenizer, seed, sample_rule="stratified", n_segment=1, is_eval=True)
    elif args.task_name == "alpaca":
        seed = 42
        val_data = AlpacaDataset(args, logger, tokenizer, seed, sample_rule="stratified", is_eval=True, data_path=args.val_data_dir)
    elif args.task_name == "law":
        seed = 42
        val_data = HFDataset(args, logger, tokenizer, seed, sample_rule="stratified", is_eval=True, data_path=args.val_data_dir)
    else:
        raise NotImplementedError(f"Unknown task {args.task_name}")
    return val_data

def get_evaluator(args, logger, model, tokenizer, output_dir_path):
    evaluator_class = Evaluator2Class(args.task_name)
    return evaluator_class(args, logger, model, tokenizer, output_dir_path)

class GPTNeoForCausalLMLossPerPoint(GPTNeoForCausalLM):
    """
        GPTNeoForCausalLM with `CrossEntropyLoss(reduction=none)` in `forward()` to obtain per-sample losses when evaluating. 
    """
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
