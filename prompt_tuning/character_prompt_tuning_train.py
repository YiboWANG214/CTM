"""
Used for training prompt template
"""
from __future__ import absolute_import, division, print_function

from transformers import BertTokenizer
from transformers import AutoModel, BertModel
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

from collections import Counter
from transformers import (
    GPT2TokenizerFast,
    BertTokenizer,
    AdamW,
    get_scheduler
)
import torch

# from prompt_tuning import BertPromptTuningLM
from character_prompt_tuning import BertPromptTuningLM
from tqdm import tqdm, trange
from tqdm import tqdm_notebook

import argparse
import csv
import logging
import os
import random
import json
import sys
import codecs
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, OrderedDict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from data import load_classification_dataset
from modeling.character_bert import CharacterBertModel
from utils.character_cnn import CharacterIndexer
from utils.data import retokenize, build_features

bert_hidden_dim = 768

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

type2hypothesis = OrderedDict({
    'brandname': ['is a brand name'],
    'product': ['is a product'],
    'features': ['is features'],
})


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None, train=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'F1':
        return {"f1": f1_score(y_true=labels, y_pred=preds)}
    else:
        raise KeyError(task_name)
    

class Config:
    # Same default parameters as run_clm_no_trainer.py in tranformers
    # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm_no_trainer.py
    num_train_epochs = 20
    weight_decay = 0.01
    learning_rate = 2e-2
    lr_scheduler_type = "linear"
    num_warmup_steps = 0
    max_train_steps = num_train_epochs
    max_seq_length=64
    train_batch_size = 8
    
    # Prompt-tuning
    # number of prompt tokens
    n_prompt_tokens = 20
    # If True, soft prompt will be initialized from vocab 
    # Otherwise, you can set `random_range` to initialize by randomization.
    init_from_vocab = True
    # random_range = 0.5


def main():
    
    args = Config()
    processors = {
        "rte": RteProcessor,
    }

    output_modes = {
        "rte": "classification",
    }
    processor = processors['rte']()
    output_mode = output_modes['rte']
    label_list = processor.get_labels()  # [0,1]\
    
    tokenizer = BertTokenizer.from_pretrained('./pretrained-models/bert-base-uncased/')
    model = BertPromptTuningLM.from_pretrained(
        './pretrained-models/general_character_bert/',
        n_tokens=args.n_prompt_tokens,
        initialize_from_vocab=args.init_from_vocab
    ).to('cuda')
    indexer = CharacterIndexer()
    
    weight = model.soft_prompt.weight
    print(len(weight), len(weight[0]))
    # print(weight)
    # print(weight[0][0])
    
    # Prepare dataset
    data = {}
    global_labels = {}
    func = load_classification_dataset
    tokenization_function = tokenizer.tokenize
    for split in ['train', 'test']:
        data[split], global_labels[split] = func(step=split, do_lower_case=True, category='brandname')
        retokenize(data[split], tokenization_function)

    # Count target labels or classes
    counter_all = Counter(
        [example.label for example in data['train']])
    counter = Counter(
        [example.label for example in data['train']])
        
    max_seq_length=64
    labels = sorted(counter_all.keys())

    logging.info("Goal: predict the following labels")
    for i, label in enumerate(labels):
        logging.info("* %s: %s (count: %s)", label, i, counter[label])

    # Input features: list[token indices] (BERT) or list[list[character indices]] (CharacterBERT)
    pad_token_id = None
    pad_token_label_id = None
    
    dataset = {}
    logging.info("Maximum sequence lenght: %s", max_seq_length)
    for split in data:
        dataset[split] = build_features(
            split=split,
            tokenizer=indexer,
            tokenizer2=tokenizer,
            examples=data[split],
            labels=labels,
            pad_token_id=pad_token_id,
            pad_token_label_id=pad_token_label_id,
            max_seq_length=max_seq_length)

    del data


    train_dataset = dataset['train']
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size)
    
    test_dataset = dataset['train']
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.train_batch_size)
    
    # inputs = tokenizer("Hello, my dog is very cute", return_tensors="pt")
    # print(inputs)
    
    # Only update soft prompt'weights for prompt-tuning. ie, all weights in LM are set as `require_grad=False`. 
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n == "soft_prompt.weight"],
            "weight_decay": args.weight_decay,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )


    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        test_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_preds = []
        for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration", position=0, leave=True)):
            # print(step)
            model.train()
            batch = tuple(t.to('cuda') for t in batch)
            input_ids, input_mask, segment_ids, label_ids, labels = batch

            outputs = model(input_ids, input_mask, labels=labels)
            loss = outputs.loss
            tr_loss += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("\ntrain loss: ", tr_loss)
        
        for step, batch in enumerate(tqdm_notebook(test_dataloader, desc="Iteration", position=0, leave=True)):
            # print(step)
            model.eval()
            batch = tuple(t.to('cuda') for t in batch)
            input_ids, input_mask, segment_ids, label_ids, labels = batch

            outputs = model(input_ids, input_mask, labels=labels)
            loss = outputs.loss
            test_loss += loss.detach()
        print("\ntest loss: ", test_loss)
    
    new_weight = list(model.soft_prompt.weight)
    print(len(new_weight), len(new_weight[0]))
    # print(new_weight)
    with open('./prompt_character_new_dic_brandname.pkl', 'wb') as f:
        pickle.dump([new_weight], f)
    

    

if __name__ == "__main__":
    main()