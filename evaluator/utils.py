import os
import pickle
from transformers import DataCollatorForLanguageModeling


def set_curriculum_str(args, method_name):
    if args.group_curriculum:
        method_name += "group_"
    if args.curriculum:
        method_name += "curriculum_"
    if args.anticurriculum:
        method_name += "anticurriculum_"
    if args.pacing_fn != "linear":
        method_name += f"{args.pacing_fn}_{args.pacing_a}_{args.pacing_b}_"
    if args.group_curriculum:
        method_name += f"mixingfrac_{args.mixing_frac}_"
    return method_name


def save_loss(loss_dict, result_path, seed, counter):
    loss_file = f"seed_{seed}_checkpoint-{counter}.pkl"
    loss_path = os.path.join(result_path, loss_file)
    with open(loss_path, "wb") as f:
        pickle.dump(loss_dict, f)
        
    return loss_path

def save_weights(weights, result_path, seed, counter):
    if weights is not None:
        weights /= sum(weights)
        weights_dict = {skill_idx: weights[skill_idx] for skill_idx in range(len(weights))}
        weights_file = f"seed_{seed}_proportions_checkpoint-{counter}.pkl"
        weights_path = os.path.join(result_path, weights_file)
        with open(weights_path, "wb") as f:
            pickle.dump(weights_dict, f)
            
            
def save_embeddings(emb_dict, result_path, seed, counter):
        emb_file = f"seed_{seed}_embeddings_checkpoint-{counter}.pkl"
        emb_path = os.path.join(result_path, emb_file)
        with open(emb_path, "wb") as f:
            pickle.dump(emb_dict, f)
            
def save_predictions(prediction_dict, result_path, seed, counter):
    prediction_file = f"seed_{seed}_predictions_checkpoint-{counter}.pkl"   
    prediction_path = os.path.join(result_path, prediction_file)
    with open(prediction_path, "wb") as f:
        pickle.dump(prediction_dict, f)
        
def save_labels(labels_dict, result_path):
    labels_path = os.path.join(result_path, "labels.pkl")
    if not os.path.exists(labels_path):
        with open(labels_path, "wb") as f:
            pickle.dump(labels_dict, f)
                 
def save_curriculum(losses, curriculum_path, seed):
    loss_file = f"seed_{seed}_curriculum.pkl"
    loss_path = os.path.join(curriculum_path, loss_file)
    with open(loss_path, "wb") as f:
        pickle.dump(losses, f)
    return loss_path

def save_skill_list(skill_list, curriculum_path, seed):
    skills_file = f"seed_{seed}_curriculum_skills.pkl"
    skills_path = os.path.join(curriculum_path, skills_file)
    with open(skills_path, "wb") as f:
        pickle.dump(skill_list, f)
    return skills_path
    

class StringDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator for samples with string data in addition to tensors."""
    def __init__(self, tokenizer, string_columns, mlm):
        super().__init__(tokenizer, mlm)
        self.string_columns = string_columns
                
    def __call__(self, examples):
        tensor_examples = [{k: v for k,v in ex.items() if k not in self.string_columns} for ex in examples]
        string_examples = [{k: v for k,v in ex.items() if k in self.string_columns} for ex in examples]
        batch = super().__call__(tensor_examples)
        counts = [len(s) for s in string_examples]
        if sum(counts) != 0:
            for col in self.string_columns:
                batch[col] = [ex[col] for ex in string_examples]
        return batch
