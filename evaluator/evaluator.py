

class AbstractEvaluator():
    def __init__(self, args, logger, model, tokenizer, output_dir_path):
        self.args = args
        self.logger = logger
        self.model = model 
        self.tokenizer = tokenizer
        self.output_dir_path = output_dir_path
        self._set_results_path()
        if args.curriculum or args.group_curriculum or args.anticurriculum:
            self._set_curriculum_path()
        pass 
    
    def _set_results_path(self):
        """Construct folder for results based on experiment configuration."""
        pass
    
    def _set_curriculum_path(self):
        """Construct folder for curriculum files (e.g., pre-trained losses over the training data)."""
        pass

    def evaluate(self, tokenized_data, counter, weights, output_idxs, train):
        """Evaluates the model on a given dataset by computing and saving the loss per sample.
        
        Args: 
        - tokenized_data: a torch dataset to evaluate the model on. This is typically the validation dataset, but can also be the training dataset when we are running a curriculum learning baseline.
        - counter: the training step at which the model is evaluated. This is used to help name the results file.
        - weights: if this is not None, we also save the weight per skill at the given training step.
        - output_idxs: if this is not None, we mask all but this index of the sample. 
        - train: if this is True, we evaluate on the training dataset.
        """
        pass 
    
    def _evaluate_train(self):
        """Evaluates the model on training data. Returns a list of losses in the same order as the dataset.
        If args.group_curriculum is set, we also return a list containing the skill of each sample in order.
        """
        pass 
        
    def _evaluate_val(self):
        """Evaluates the model on validation data. Returns a dictionary mapping from each skill to the list of losses corresponding to samples associated with that skill."""
        pass 
