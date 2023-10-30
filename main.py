import os
from transformers import GPT2TokenizerFast
import wandb
from transformers import set_seed
import argparse
from datetime import datetime
from utils import get_trainer, get_logger, get_val_dataset, get_evaluator, make_output_dir, GPTNeoForCausalLMLossPerPoint
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Skill-It data selection"
    )
    # data loading arguments
    parser.add_argument(
        "--task_name",
        type=str,
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        help="Directory from which to load training data",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        help="Directory from which to load validation data",
        default=None,
    )
    parser.add_argument(
        "--debug_val",
        action="store_true",
        help="If set to true, use a smaller validation dataset ",
    )
    parser.add_argument(
        "--slicer",
        nargs="+",
        default=["task_category"],
        help="For Natural Instructions, this is the attribute name along which skills are defined. We use task categories as skills, except in Spanish question generation, where slicer=['task_category', 'input_language', 'output_language']."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/",
        help="Directory where all results and logs are stored."
    )
    parser.add_argument(
        "--dev_split_path",
        type=str,
        help="Path of .pkl file containing dictionary for NI/Alpaca training samples that belong to the validation set",
        default="./aux_data/dev_split_map.pkl",
    )
    # Natural Instructions-specific arguments
    parser.add_argument(
        "--xlingual",
        action="store_true",
        help="Only used for NI. If true, use crosslingual split of tasks (see https://github.com/allenai/natural-instructions/tree/master/splits/xlingual)"
    )
    parser.add_argument(
        "--ni_test",
        action="store_true",
        help="Only used for NI OOD setting. If true, use test split of tasks for evaluation"
    )
    parser.add_argument(
        "--ni_task_info_path",
        type=str,
        default="./aux_data/ni_default_task_info.pkl",
        help="Path to pkl file containing metadata about NI (e.g. task name, task category, input language, output language).",
    )
    parser.add_argument(
        "--stop_when_done",
        action="store_true",
    )
    parser.add_argument(
        "--one_sample_per_window",
        action="store_true",
        default=None,
    )
    # general data sampling arguments
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n_select",
        type=int,
        help="Number of training points to select. Can use this or max_steps to specify duration of training.",
        default=0,
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Curriculum learning using scores from the initial model in increasing order.",
    )
    parser.add_argument(
        "--anticurriculum",
        action="store_true",
        help="Anticurriculum learning using scores from the initial model in decreasing order.",
    )
    parser.add_argument(
        "--group_curriculum",
        action="store_true",
        help="Group curriculum learning approach based on https://arxiv.org/pdf/2205.09898.pdf",
    )
    parser.add_argument(
        "--mixing_frac",
        type=float,
        default=0,
        help="Group curriculum learning mixing rate of previous datasets",
    )
    parser.add_argument(
        "--curriculum_analysis_only",
        action="store_true",
        help="Runs curriculum learning baseline, but only returns analysis (a sample's skill and order in training).",
    )
    parser.add_argument(
        "--pacing_fn",
        type=str,
        default="linear",
        choices=["linear", "log"],
        help="Pacing function for curriculum learning.",
    )
    parser.add_argument(
        "--pacing_a",
        type=float,
        default=None,
        help="Pacing function parameter for log pacing.",
    )
    parser.add_argument(
        "--pacing_b",
        type=float,
        default=None,
        help="Pacing function parameter for log pacing.",
    )
    parser.add_argument(
        "--slice_list",
        type=str,
        nargs="+",
        help="1) Path to file (.txt) with list of skills to filter on (newline separated), or 2) the direct list of skills to filter. If a skill is defined along multiple axes, they are flattened, e.g., ['question_generation', 'spanish', 'spanish', 'question_answering', 'spanish', 'spanish'] consists of two skills.",
        default=None,
    )
    parser.add_argument(
        "--exclude_slice",
        type=str,
        help="if this is set, we exclude this slice from the training dataset. This is useful for doing one versus all experiments when the number of skills is large.",
        default=None,
    )
    parser.add_argument(
        "--filter_val_skills",
        action="store_true",
        help="If true, the validation dataset will also only consist of skills in slice_list. This is useful in the Natural Instructions fine-tuning settings."
    )
    parser.add_argument(
        "--sample_rule",
        type=str,
        default=None,
        help="Strategy for sampling after filtering. `stratified` means 1/k probability per skill, while `mixture` enables using a custom list of proportions.",
    )
    parser.add_argument(
        "--target_mask",
        type=str,
        nargs="+",
        default=None,
        help="List of 0/1s to indicate which of the skills in the slice_list are the target skills.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of skills. Needs to be set for synthetics, but for real datasets this can be inferred from the data.",
    )
    parser.add_argument(
        "--lego_graph",
        type=str,
        default=None,
        help="True LEGO graph among variables. The typical LEGO construction is a 'chain'.",
        choices=[None, "disconnected", "tree"]
    )
    parser.add_argument(
        "--n_segment",
        type=int,
        default=1,
        help="Number of training segments in synthetics to specify skill mixtures over. Default is 1.",
    )
    parser.add_argument(
        "--proportions",
        type=float,
        nargs="+",
        default=None,
        help="List of proportions to sample with (static). Does not need to add up to 1. The length should be equal to slice_list * n_segment if slice_list is set, otherwise k * n_segment.",
    )
    parser.add_argument(
        "--proportions_schedule",
        type=float,
        nargs="+",
        default=None,
        help="List of proportions to sample with. Does not need to add up to 1. If this is set, the training procedure will be divided into len(proportions_schedule)/(len(slice_list) or k) segments of equal length.",
    )
    parser.add_argument(
        "--proportions_file",
        type=str,
        default=None,
        help="Path to .npy file with proportions, doesn't need to add up to 1.",
    )
    parser.add_argument(
        "--segment_proportions",
        type=float,
        nargs="+",
        default=None,
        help="List of proportions to indicate how long each segment is. Does not need to add up to 1. Only supported in synthetics.",
    )
    # Skill-It algorithm arguments
    parser.add_argument(
        "--mw",
        action="store_true",
        help="Use 'multiplicative weights' (e.g., Skill-It online sampling) algorithm"
    )
    parser.add_argument(
        "--update_steps",
        type=int,
        default=None,
        help="How often to update multiplicative weights"
    )
    parser.add_argument(
        "--eta",
        type=float,
        help="eta parameter for weight update"
    )
    parser.add_argument(
        "--eta_schedule",
        action="store_true",
        help="If true, we increase eta as a function of loss: eta = eta_0 / sum loss. With static eta, as losses get smaller the magnitude of weight updates also get smaller."
    )
    parser.add_argument(
        "--normalize_loss",
        action="store_true",
        help="If true, we normalize weight update by original loss at time 0, so that the skill's weight is based on relative loss"
    )
    parser.add_argument(
        "--initialize_loss",
        action="store_true",
        help="Add an initial loss L_0 equal to uniform (normalized). This is used for smoother transition from initial mixture to the first mixture"
    )
    parser.add_argument(
        "--mw_window",
        type=int,
        default=3,
        help="Look-back window for weight update."
    )
    parser.add_argument(
        "--mw_prior",
        type=float,
        nargs="+",
        default=None,
        help="Sampling prior throughout the run"
    )
    parser.add_argument(
        "--mw_init",
        type=float,
        nargs="+",
        default=None,
        help="Initialization for sampling distribution"
    )
    parser.add_argument(
        "--graph",
        type=int,
        nargs="+",
        default=None,
        help="Skills graph (flattened)"
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default=None,
        help="Path to .npy file containing skills graph"
    )
    parser.add_argument(
        "--ignore_lone_nodes",
        action="store_true",
        help="If true, we set skills who have no ingoing/outgoing edges in the skills graph to have weight 1/k (uniform sampling)"
    )
    parser.add_argument(
        "--dynamic_lambda",
        action="store_true",
        help="If true, we decrease the edge weights of the skills graph as time goes on. This is worth setting when we believe that dependencies among skills grow weaker as skills are learned."
    )
    # training arguments
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--n_epochs",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--lr",
        default=5e-5,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of steps to train for. Overrides n_epochs.",
    )
    parser.add_argument(
        "--model_name",
        default="EleutherAI/gpt-neo-125M",
        type=str,
        help="Model to continually pre-train/fine-tune",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
    )
    # evaluation args
    parser.add_argument(
        "--num_ckpts",
        help="Number of checkpoints to evaluate the model at.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--get_embs",
        action="store_true",
        help="If true, we save last layer embeddings at every checkpoint."
    )
    parser.add_argument(
        "--session_id",
        default=None,
        help="Name of specific folder (generally MonthDateYear) to save results to."
    )

    args = parser.parse_args()
    return args

def main():
    run_id = datetime.now().strftime("%m%d%Y")
    _ = wandb.init(mode="disabled")
    args = parse_args()
    output_dir_path = make_output_dir(args.output_dir, args.session_id, run_id)
    
    logger = get_logger(output_dir_path)
    logger.info(args)

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForCausalLMLossPerPoint.from_pretrained(args.model_name).cuda()

    logger.info("Constructing validation data.")
    validation_data = get_val_dataset(args, logger, tokenizer)
    
    evaluator = get_evaluator(args, logger, model, tokenizer, output_dir_path)    
    logger.info("Training model!")
    trainer = get_trainer(args)   
    trainer.train(args, logger, tokenizer, model, validation_data, evaluator)


if __name__ == "__main__":
    main()
