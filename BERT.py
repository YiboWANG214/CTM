"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

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
import matplotlib.pyplot as plt
import torch
from collections import defaultdict, OrderedDict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from tqdm import tqdm_notebook

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForMaskedLM, AutoModelForSequenceClassification
from transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers.optimization import AdamW
from transformers import RobertaTokenizer
from transformers.optimization import AdamW
from transformers import RobertaModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification #RobertaModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
# from pytorch_transformers import *

from preprocess_emotion import evaluate_emotion_zeroshot_TwpPhasePred

bert_hidden_dim = 768
# pretrain_model_dir = 'roberta-large'
# pretrain_model_dir = 'model_save_title'
pretrain_model_dir = 'bert-base-uncased'
# pretrain_model_dir = 'model_save_bert_finetuned'
# pretrain_model_dir = 'deepset/roberta-base-squad2'
# pretrain_model_dir = 'madlag/bert-large-uncased-mnli'
# pretrain_model_dir = 'bert-large-uncased-mnli'
# pretrain_model_dir = 'helboukkouri/character-bert'
# pretrain_model_dir = 'textattack/bert-base-uncased-MNLI'


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



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
        if train==True:
            self.label_id = label_id


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

    def get_examples_title_train(self, filename, size_limit_per_type):
        readfile = codecs.open(filename, 'r', 'utf-8')

        line_co = 0
        exam_co = 0
        examples = []
        label_list = []

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
            guid = "train-" + str(exam_co)
            text_b = token
            label = type_list[0]
            # print(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            exam_co += 1

            if exam_co > size_limit_per_type:
                break

        readfile.close()
        print('train loaded size:', exam_co)
        return examples, set()
        
    def get_examples_title_test(self, filename, seen_types, label, limit_size):
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co = 0
        exam_co = 0
        examples = []
        count = 0

        hypo_seen_str_indicator = []
        hypo_2_type_index = []

        dic = {'brandname':0, 'product':1, 'features':2}
        gold_label_list = []
        gold_label_list2 = []
        for row in readfile:
            count += 1
            if count > limit_size:
                break
            line = row.strip()
            line = line.split('  ')
            if len(line) >= 2:
                type_index = line[0].split(' ')[0].strip()
                token = ' '.join(line[0].split(' ')[1:]).strip()
                gold_label_list.append(dic[type_index])
                gold_label_list2.append(type_index)

                guid = "test-" + str(exam_co)
                text_a = line[1]
                text_b = token
                label = type_index
                # print(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                exam_co += 1

                line_co += 1
                if line_co % 1000 == 0:
                    print('loading test size:', line_co)
        return examples, gold_label_list, gold_label_list2, hypo_seen_str_indicator, hypo_2_type_index

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
        return ['brandname', 'product', 'features']

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
                                 tokenizer, output_mode, train=True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    premise_2_tokenzed = {}
    hypothesis_2_tokenzed = {}
    list_2_tokenizedID = {}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = premise_2_tokenzed.get(example.text_a)
        if tokens_a is None:
            tokens_a = tokenizer.tokenize(example.text_a)
            premise_2_tokenzed[example.text_a] = tokens_a

        tokens_b = premise_2_tokenzed.get(example.text_b)
        if tokens_b is None:
            tokens_b = tokenizer.tokenize(example.text_b)
            hypothesis_2_tokenzed[example.text_b] = tokens_b

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens_A = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids_A = [0] * len(tokens_A)
        tokens_B = tokens_b + ["[SEP]"]
        segment_ids_B = [1] * (len(tokens_b) + 1)
        tokens = tokens_A + tokens_B
        segment_ids = segment_ids_A + segment_ids_B

        input_ids_A = list_2_tokenizedID.get(' '.join(tokens_A))
        if input_ids_A is None:
            input_ids_A = tokenizer.convert_tokens_to_ids(tokens_A)
            list_2_tokenizedID[' '.join(tokens_A)] = input_ids_A
        input_ids_B = list_2_tokenizedID.get(' '.join(tokens_B))
        if input_ids_B is None:
            input_ids_B = tokenizer.convert_tokens_to_ids(tokens_B)
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

        label_id = None
        if train:
            label_id = label_map[example.label]

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
                          label_id=label_id,
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


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()



    processors = {
        "rte": RteProcessor,
    }

    output_modes = {
        "rte": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()  # [0,1]
    print(label_list)
    num_labels = len(label_list)
    

    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(pretrain_model_dir,
                                                                num_labels=3,
                                                                force_download=True,
                                                                )


    if args.fp16:
        model.half()
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate)
                          
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_test_unseen_acc = 0.0
    max_dev_unseen_acc = 0.0
    max_dev_seen_acc = 0.0
    max_overall_acc = 0.0
    test_f1_list = []
    test_loss_list = []
    train_f1_list = []
    train_loss_list = []
    
    if args.do_train:
        train_examples, seen_types = processor.get_examples_title_train('./title/train_new.txt', 100000)

        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)

        '''load test set'''
        test_examples, test_label_list, test_label_list2, test_hypo_seen_str_indicator, test_hypo_2_type_index = processor.get_examples_title_test(
            './title/test_new.txt', seen_types, True, 50000)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer, output_mode, True)

        test_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        test_all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        test_all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

        if output_mode == "classification":
            test_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        elif output_mode == "regression":
            test_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.float)

        test_data = TensorDataset(test_all_input_ids, test_all_input_mask, test_all_segment_ids, test_all_label_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.train_batch_size)
        
        
        '''load train set for evaluation'''

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            train_preds = []
            for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration", position=0, leave=True)):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = model(input_ids, input_mask)[0]
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            train_loss_list.append(tr_loss)
                    
            logger.info("===========start inference============")
            model.eval()
    
            test_tr_loss = 0
            nb_test_steps = 0
            preds = []
            print('Inferencing...')
            for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
    
                with torch.no_grad():
                    logits = model(input_ids, input_mask)[0]
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                test_tr_loss += loss.item()
                # print(logits.shape)
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
    
            # eval_loss = eval_loss / nb_eval_steps
            print(preds[0].shape)
            preds = preds[0]
            print("!", preds)
            preds = softmax(preds,axis=1)
            test_probs = []
            for pred in preds:
                pred = list(pred)
                my_max = max(pred)
                my_index = pred.index(my_max)
                test_probs.append(my_index)                
                
            print(test_label_list[:10])
            print(classification_report(test_label_list, test_probs, digits=4))
            

if __name__ == "__main__":
    main()
