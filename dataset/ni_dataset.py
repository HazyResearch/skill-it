import numpy as np
import pickle
import os
import json
import random
from itertools import cycle
from collections import OrderedDict, defaultdict
import torch
from torch.utils.data import IterableDataset
import pandas as pd


from .utils import get_filter_skills, get_weights_from_graph

from tqdm import tqdm

from .dataset import AbstractDataset


# NOTE: why do uniform over tasks? Why not uniform over samples?
# it is because we do not know the overall # of samples (after shoving them all into context windows)
# I guess we can do a version where we do one sample per window and then do uniform over that? 
# hierarchy is sample -> task -> task category 
# also because the data sampler selects a task randomly and then selects a point from that task randomly.
# it is not possible to purely randomly select a point? look into this . edit: it's possible just annoying.


class NIDataset(AbstractDataset):
    """
    Sampled dataset where the base dataset is Natural Instructions from https://github.com/allenai/natural-instructions.

    """

    def __init__(
        self,
        args,
        logger,
        tokenizer,
        seed,
        sample_rule,
        is_eval,
        data_path,
        dev_split_path=None,
        ni_task_info_path=None,
    ):
        """
        Construct the NaturalInstructionsDataset.

        Args:
        - tokenizer
        - data_path
        - dev_split_path: natural instructions does not have the same train tasks and test tasks. To test the model on the training tasks, we split each
            train task into 100 development samples while the rest is training data. dev_split_path is the path to the dictionary file mapping from task
            name to the indices of the development samples.
        - context_length
        - selection_seed
        """
        super().__init__(
            args, logger, tokenizer, seed, sample_rule, is_eval, data_path
        )

        self.debug_val = args.debug_val
        self.stop_when_done = args.stop_when_done
        self.one_sample_per_window = args.one_sample_per_window
        self.slicer = args.slicer

        # load in dev split
        if dev_split_path is None:
            dev_split_path = args.dev_split_path 
        with open(dev_split_path, "rb") as f:
            self.dev_split = pickle.load(f)

        # get all the training task paths
        self.train_splits = []
        splits_file = (
            "splits/xlingual/train_tasks.txt"
            if args.xlingual
            else "splits/default/test_tasks.txt" 
            if args.ni_test and self.is_eval
            else "splits/default/train_tasks.txt"
        )
        with open(os.path.join(self.data_path, splits_file)) as f:
            for line in f:
                if line.strip() == "":
                    continue
                self.train_splits.append(line.strip())  # just get the full task names

        # now open the train task info DF
        if ni_task_info_path is None:
            ni_task_info_path = args.ni_task_info_path
        if not os.path.exists(ni_task_info_path):
            raise ValueError("Invalid path for ni_task_info.")

        self.train_task_info = pd.read_pickle(ni_task_info_path)

        # filter out bad datasets
        if args.xlingual:
            self.train_task_info = self.train_task_info.loc[
                self.train_task_info.n_samples >= 200
            ]
        elif args.ni_test and self.is_eval:
            pass 
        else:
            self.train_task_info = self.train_task_info.loc[
                self.train_task_info.n_samples >= 100
            ]

        self.k = args.k
        
        self.set_skills(args)
        self.set_proportions(args, args.proportions_file if args.proportions_file is not None else args.proportions)

    def set_skills(self, args):
        
        slices, _ = get_filter_skills(args.slice_list, args.exclude_slice, self.k)
        if slices is not None and (not self.is_eval or (self.is_eval and args.filter_val_skills)):
            self.skills = np.array(slices)
        else:
            if len(self.slicer) > 1:
                self.skills = np.array(sorted(self.train_task_info.groupby(self.slicer).size().reset_index()[self.slicer].values))
            else:
                self.skills = np.array(sorted(self.train_task_info[self.slicer[0]].unique()))
            
        self.k = len(self.skills)
        self.logger.info(f"Remaining {self.k} skills:\n{self.skills}")
        if len(self.slicer) > 1:
            self.skills = [tuple(i) for i in self.skills]
            self.skills_to_tasks = {s: self.train_task_info.loc[self.train_task_info[self.slicer].apply(tuple, axis=1) == s].long_task.values for s in self.skills}
        else:
            subset_train_task_info = self.train_task_info.loc[
                self.train_task_info[self.slicer[0]].isin(self.skills)
            ].sort_values(by=self.slicer[0])
            self.skills_to_tasks = {s: subset_train_task_info.loc[subset_train_task_info[self.slicer[0]] == s].long_task.values for s in sorted(self.skills)}    

    def set_proportions(self, args, proportions):
        self.tasks_to_p = {}

        if self.sample_rule == "mixture":
            if proportions is not None and ".npy" in proportions:
                proportions = np.load(proportions)
            elif args.graph is not None or args.graph_path is not None and not args.mw:
                proportions = get_weights_from_graph(args)  
                
            if proportions is not None:             
                proportions = np.array(proportions)
                proportions /= sum(proportions)
                self.proportions = proportions 
                assert len(self.proportions) == self.k, f"Length of proportions is {len(self.proportions)} but k is {self.k}"

                self.logger.info(f"Setting skill proportions:\n{self.proportions}")

                for skill, p in zip(self.skills, self.proportions):
                    tasks = self.skills_to_tasks[skill]
                    task_p = p / len(tasks)
                    for task in tasks:
                        self.tasks_to_p[task] = task_p
            else:
                self.logger.info(f"Performing uniform sampling over tasks.")
                n_tasks = len(self.train_task_info.long_task.values)
                self.tasks_to_p = {task: 1.0 / n_tasks for task in self.train_task_info.long_task.values}

        elif self.sample_rule == "stratified":
            self.logger.info(f"Performing stratified sampling over skills.")
            for skill in self.skills:
                tasks = self.skills_to_tasks[skill]
                task_p = 1 / (len(tasks) * self.k)
                for task in tasks:
                    self.tasks_to_p[task] = task_p
        else:
            self.logger.info(f"Performing uniform sampling over tasks.")
            all_tasks = [item for sublist in list(self.skills_to_tasks.values()) for item in sublist]
            n_tasks = len(all_tasks)
            self.tasks_to_p = {task: 1.0 / n_tasks for task in all_tasks}


    def get_tokenized_dataset(self):
        self.task_dict = OrderedDict()
        for task_name in self.tasks_to_p:
            task_path = os.path.join(self.data_path, "tasks", task_name + ".json")
            with open(task_path) as f:
                task = json.load(f)
            output_space = set()
            is_classification = True
            if self.is_eval:
                # only keep dev data
                if self.debug_val:
                    # we just sample one point per task for debugging
                    task["Instances"] = [task["Instances"][self.dev_split[task_name][0]]]
                else:
                    task["Instances"] = [obj for i, obj in enumerate(task["Instances"]) if i in self.dev_split[task_name]]
            else:
                # remove dev data
                task["Instances"] = [obj for i, obj in enumerate(task["Instances"]) if i not in self.dev_split[task_name]]

            for instance in task["Instances"]:
                output_space.add(instance["output"][0])
                if len(output_space) > 10:
                    is_classification = False
                    break

            task["IsClassification"] = is_classification
            task["OutputSpace"] = (
                sorted(list(output_space)) if is_classification else None
            )
            self.task_dict[task_name] = task

        
        dataset = NIStreamDataset(
            self.task_dict,
            self.tasks_to_p,
            self.tokenizer,
            self.context_length,
            self.train_task_info,
            self.stop_when_done,
            self.slicer,
            self.one_sample_per_window,
            self.is_eval,
            self.seed,
        )
        
        if not self.is_eval:
            return dataset
        else:
            return dataset, None


class NIStreamDataset(IterableDataset):
    """
    An iterable Natural Instructions dataset. Generates samples by sampling at the task level and then
    fitting the task definition and as many formatted instances of the task as possible in the context window.
    """

    def __init__(
        self,
        task_dict,
        tasks_to_p,
        tokenizer,
        context_length,
        train_task_info,
        stop_when_done,
        slicer,
        one_sample_per_window,
        is_eval,
        seed,
    ):
        """
        Construct an NIStreamDataset.

        Args:
        - tasks: dictionary mapping from task name to a dictionary containing task instances.
        - classification_tasks: dictionary with classification tasks only.
        - tokenizer: for tokenizing the data.
        - context_length: max context length of samples.
        - sample_tasks: list of tasks that we sample from. Might not be all tasks if we filter the dataset. If none, is all tasks.
        - sample_probs: list of probabilities to sample proportional to. len(sample_probs) == len(sample_tasks)
        - eval: if we are constructing the training or validation dataset.
        - no_replace: if True, ensures that we cycle through all the instances of a task before sampling already seen instances.
        """
        self.task_dict = task_dict
        self.tasks_to_p = tasks_to_p
        self.train_task_info = train_task_info
        self.iter_count = 0
        self.is_eval = is_eval
        self.stop_when_done = stop_when_done
        self.one_sample_per_window = one_sample_per_window
        self.slicer = slicer 
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.it = None
        self.seed = seed 
                
        # for formatting the instances
        if self.is_eval:
            # we want deterministic validation dataset
            self.input_prefixs = ["Input: "]
            self.output_prefixs = ["Output: "]
            self.sample_splitters = ["\n\n"]
            self.answer_splitters = ["\n"]
        else:
            self.input_prefixs = [
                "Input: ",
                "Given: ",
                "Context: ",
                "Example: ",
                "Question: ",
                "",
                "",
                "",
                "",
                "",
            ]
            self.output_prefixs = [
                "Output: ",
                "Output: ",
                "Ans: ",
                "A: ",
                "Answer: ",
                "Label: ",
                "Label: ",
            ]
            self.sample_splitters = [
                "\n\n",
                "\n",
                "\n\n",
                "\n\n\n",
                "\n###\n",
                "\n---\n",
            ]
            self.answer_splitters = ["\n", "\n", "\n\n"]

        # keep track of instances per task sampled, and tasks whose instances have all been sampled
        self.samples_so_far = defaultdict(list)
        self.tasks_so_far = set()
        
        
    def reset_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def state_dict(self):
        return {"iter_count": self.iter_count}

    def _load_state_dict(self, state_dict):
        try:
            self.iter_count = state_dict["iter_count"]
        except:
            print("cannot load ni states.")

    def sample_text_from_task(self, task_name):
        """
        Constructs one sample from the given task. Format:

        Task definition
        Input: x1_input Output: x1_output
        Input: x2_input Output: x2_output
        .
        .
        .

        until the context_length is reached. If no_replace is True, there are no duplicate instances.

        Args:
        - task_name: name of task to sample from
        """
        task = self.task_dict[task_name]
        is_classification = task["IsClassification"]
        output_space = task["OutputSpace"]

        sample_splitter = random.choice(self.sample_splitters)
        answer_splitter = random.choice(self.answer_splitters)
        text_def = random.choice(task["Definition"] + task["Definition"] + [""]).strip()
        text_input = random.choice(self.input_prefixs)
        text_output = random.choice(self.output_prefixs)

        # construct task definition
        if is_classification and random.random() < 0.5:
            text_def += "\nPossible labels:"
            for i, possible_output in enumerate(output_space):
                text_def += f"\n{i+1}. {possible_output}"
            text_def += "\n"

        # build up text_context to tokenize
        text_context = text_def
        # boolean flag for if we have sampled all instances from the task
        task_done = False
        while True:
                # only pick instances that are not already sampled
            instance_idx = np.random.choice(
                np.setdiff1d(
                    np.arange(len(task["Instances"])),
                    np.array(self.samples_so_far[task_name]),
                )
            )
                
            instance = task["Instances"][instance_idx]
            self.samples_so_far[task_name].append(instance_idx)

            
            text_context += (
                sample_splitter
                + text_input
                + instance["input"]
                + answer_splitter
                + text_output
                + random.choice(instance["output"])
            )
            
            if self.one_sample_per_window:
                input_ids = self.tokenizer(text_context.strip(), padding="max_length")['input_ids']
                self.tasks_so_far.add(task_name)
                self.samples_so_far[task_name] = ([])
                task_done = True 
                break
            

            if set(np.arange(len(task["Instances"]))) == set(
                self.samples_so_far[task_name]
            ):
                
                # if we have now sampled all instances in a task, tokenize the data and update samples_so_far and tasks_so_far
                input_ids = self.tokenizer(text_context.strip(), padding="max_length")["input_ids"]
                self.tasks_so_far.add(task_name)
                self.samples_so_far[task_name] = ([])  
                # reset this cache. Basically, we want to ensure that all samples are picked before we duplicate them
                task_done = True
            else:
                input_ids = self.tokenizer(text_context.strip())["input_ids"]
            if len(input_ids) >= self.context_length:
                # will run this if we've used all the points in a task
                break

        input_ids = input_ids[: self.context_length]
        input_ids = torch.tensor(input_ids).long()
        return task_done, input_ids

    def get_eval_sequence(self):
        """
        For creating the validation dataset, we iterate through all tasks and all instances of each task.
        Yields both the task name and input_ids so that we can compute performance per validation task.
        """
        for task_name in self.tasks_to_p.keys():
            task_done = False
            while not task_done:
                task_done, input_ids = self.sample_text_from_task(task_name)
                self.iter_count += 1
                task_info = self.train_task_info.loc[
                    self.train_task_info.long_task == task_name
                ]
                info_dict = {slicer_dim: task_info[slicer_dim].values[0] for slicer_dim in self.slicer}
                ret_dict = {"task": task_name, "input_ids": input_ids}
                ret_dict.update(info_dict)
                yield ret_dict
                

    def get_sequence(self):
        """
        To generate a sequence, we first select a task at random. In particular, we skew towards classification tasks, and also sample proportional to self.sample_probs if specified.
        Then, we call sample_text_from_task on the selected task.
        """
        while True:
            if self.stop_when_done:
                sorted_categories = sorted(self.train_task_info[self.slicer].unique())
                for category in sorted_categories:
                    tasks = self.train_task_info.loc[
                        self.train_task_info[self.slicer] == category
                    ].long_task.values
                    if set(tasks).issubset(self.tasks_so_far):
                        print(f"Have run out of samples for task category {category}")
                        raise StopIteration()
            task_name = np.random.choice(
                np.array(list(self.tasks_to_p.keys())), p=np.array(list(self.tasks_to_p.values()))
            )
            
            
            # then pick a random sample from that task. Note that we have already removed the dev data from the task.
            _, input_ids = self.sample_text_from_task(task_name)
            
            task_info = self.train_task_info.loc[
                    self.train_task_info.long_task == task_name
            ]
            self.iter_count += 1
            info_dict = {slicer_dim: task_info[slicer_dim].values[0] for slicer_dim in self.slicer}
            ret_dict = {"task": task_name, "input_ids": input_ids}
            ret_dict.update(info_dict)
            yield ret_dict


    def get_stream(self):
        if self.is_eval:
            return self.get_eval_sequence()
        else:
            return cycle(self.get_sequence())

    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()

        if not self.is_eval:
            for i in range(self.iter_count):
                next(self.it)

        return self.it

