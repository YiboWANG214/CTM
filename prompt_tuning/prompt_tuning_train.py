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

from transformers import (
    GPT2TokenizerFast,
    BertTokenizer,
    AdamW,
    get_scheduler
)
import torch

from prompt_tuning import BertPromptTuningLM
# from character_prompt_tuning import BertPromptTuningLM
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

    def get_examples_title_train(self, filename, size_limit_per_type, category):
        readfile = codecs.open(filename, 'r', 'utf-8')

        line_co = 0
        exam_co = 0
        examples = []
        label_list = []
        
        golden_label_list = []
        dic = dict()
        dic['brandname'] = 0
        dic['product'] = 1
        dic['features'] = 2
        
        # history_types = set()
        history_types = set(['brandname', 'product', 'features'])
        for line in readfile:

            category_token = line.split('  ')[0]
            type_list = [category_token.split(' ')[0]]
            token = ' '.join(category_token.split(' ')[1:])
            text = ' '.join(line.split('  ')[1:])

            neg_types = history_types - set(type_list)

            if len(neg_types) > 1:
                sampled_type_set = random.sample(neg_types, 1)
            else:
                # print('?')
                continue

            '''pos pair'''
            text_a = text
            if type_list[0] == category:
                golden_label_list.append(dic[type_list[0]])
                guid = "train-" + str(exam_co)
                # text_b = token
                text_b = token
                label = category  # if line[0] == '1' else 'not_entailment'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=category))
                exam_co += 1

        readfile.close()
        print('train loaded size:', exam_co)
        return examples, golden_label_list, set()
        
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


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 bert_tokenizer, output_mode, train=True):
    """Loads a data file into a list of `InputBatch`s."""

    premise_2_tokenzed = {}
    hypothesis_2_tokenzed = {}
    list_2_tokenizedID = {}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = premise_2_tokenzed.get(example.text_a)
        if tokens_a is None:
            tokens_a = bert_tokenizer.tokenize(example.text_a)
            premise_2_tokenzed[example.text_a] = tokens_a

        tokens_b = premise_2_tokenzed.get(example.text_b)
        if tokens_b is None:
            tokens_b = bert_tokenizer.tokenize(example.text_b)
            hypothesis_2_tokenzed[example.text_b] = tokens_b

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens_A = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids_A = [0] * len(tokens_A)
        tokens_B = tokens_b + ["[SEP]"]
        segment_ids_B = [1] * (len(tokens_B))
        tokens = tokens_A + tokens_B
        segment_ids = segment_ids_A + segment_ids_B
        
        input_ids_A = list_2_tokenizedID.get(' '.join(tokens_A))
        if input_ids_A is None:
            input_ids_A = bert_tokenizer.convert_tokens_to_ids(tokens_A)
            list_2_tokenizedID[' '.join(tokens_A)] = input_ids_A
        input_ids_B = list_2_tokenizedID.get(' '.join(tokens_B))
        if input_ids_B is None:
            input_ids_B = bert_tokenizer.convert_tokens_to_ids(tokens_B)
            list_2_tokenizedID[' '.join(tokens_B)] = input_ids_B
        input_ids = input_ids_A + input_ids_B

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          train=train
                          ))
    return features


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
    learning_rate = 3e-2
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
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Initialize GPT2LM with soft prompt
    model = BertPromptTuningLM.from_pretrained(
        "bert-base-uncased",
        n_tokens=args.n_prompt_tokens,
        initialize_from_vocab=args.init_from_vocab
    ).to('cuda')
    
    weight = model.soft_prompt.weight
    print(len(weight), len(weight[0]))
    print(weight)
    
    # Prepare dataset
    train_examples, golden_label_list, seen_types = processor.get_examples_title_train('./data/train_new.txt', 100000, 'brandname')
    train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    test_examples, test_golden_label_list, test_seen_types = processor.get_examples_title_train('./data/test_new.txt', 100000, 'brandname')
    test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    all_test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_test_input_ids, all_test_input_mask, all_test_segment_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.train_batch_size)

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
            input_ids, input_mask, segment_ids = batch

            outputs = model(input_ids, input_mask, labels=input_ids)
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
            input_ids, input_mask, segment_ids = batch

            outputs = model(input_ids, input_mask, labels=input_ids)
            loss = outputs.loss
            test_loss += loss.detach()
        print("\ntest loss: ", test_loss)
    
    new_weight = list(model.soft_prompt.weight)
    print(len(new_weight), len(new_weight[0]))
    # print(new_weight)
    with open('./prompt_new_dic_brandname.pkl', 'wb') as f:
        pickle.dump([new_weight], f)
    

    

if __name__ == "__main__":
    main()