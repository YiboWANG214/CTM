
from __future__ import absolute_import, division, print_function

from transformers import BertTokenizer, BertConfig
from transformers import AutoModel, BertModel
from modeling.character_bert import CharacterBertModel
from utils.character_cnn import CharacterIndexer

from data import load_classification_dataset, load_sequence_labelling_dataset

import datetime
from collections import Counter
from utils.misc import set_seed
from utils.data import retokenize, build_features
from utils.training import train, evaluate

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
# from transformers.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead, BertPreTrainedModel


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
import matplotlib.pyplot as plt
import torch
from collections import defaultdict, OrderedDict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

import torch.utils.data as data
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
from transformers.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead, BertPreTrainedModel

from transformers.optimization import AdamW
from transformers import RobertaTokenizer
from transformers.optimization import AdamW
from transformers import RobertaModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification #RobertaModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

bert_hidden_dim = 1024

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

type2hypothesis = OrderedDict({
    'brandname': ['is a band name, which is a type of things manufactured by a particular company under a particular name.'],
    'product': ['is a product, which is an article or substance that is manufactured or refined for sale.'],
    'features': ['is a feature, which is a distinctive attribute or aspect of something.'],
    # 'brandname': ['is a brand name'],
    # 'product': ['is a product'],
    # 'features': ['is a feature'],
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

            sampled_type_set = random.sample(neg_types, 1)

            '''pos pair'''
            text_a = text
            for hypo in type_list:
                golden_label_list.append(dic[hypo])
                guid = "train-" + str(exam_co)
                # text_b = token
                text_b = token + ' ' + type2hypothesis[hypo][0]
                label = 'entailment'  # if line[0] == '1' else 'not_entailment'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                exam_co += 1

            # '''neg pair'''
            for hypo in sampled_type_set:
                guid = "train-" + str(exam_co)
                text_b =  token + ' ' + type2hypothesis[hypo][0]
                label = 'not_entailment'  # if line[0] == '1' else 'not_entailment'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                exam_co += 1
            if exam_co > size_limit_per_type:
                break

        readfile.close()
        print('train loaded size:', exam_co)
        return examples, golden_label_list, set()
        
    def get_examples_title_test(self, filename, limit_size):
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co = 0
        exam_co = 0
        examples = []
        count = 0
        
        dic = dict()
        dic['brandname'] = 0
        dic['product'] = 1
        dic['features'] = 2

        hypo_2_type_index = []
        '''notice that noemo hasnt be set as unseen, we treat it in evaluation part'''

        gold_label_list = []
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
                for type, hypo_list in type2hypothesis.items():
                    if type == type_index:
                        '''pos pair'''
                        for hypo in hypo_list:
                            guid = "test-" + str(exam_co)
                            text_a = line[1]
                            text_b = token + ' ' + hypo
                            label = 'entailment'  # if line[0] == '1' else 'not_entailment'
                            examples.append(
                                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                            exam_co += 1
                    else:
                        '''neg pair'''
                        for hypo in hypo_list:
                            guid = "test-" + str(exam_co)
                            text_a = line[1]
                            text_b = token + ' ' + hypo
                            label = 'not_entailment'  # if line[0] == '1' else 'not_entailment'
                            examples.append(
                                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                            exam_co += 1
                line_co += 1
                if line_co % 1000 == 0:
                    print('loading test size:', line_co)
        return examples, gold_label_list, hypo_2_type_index          
        

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

        label_id = None
        if train:
            if output_mode == "classification":
                label_id = label_map[example.label]
            else:
                raise KeyError(output_mode)

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



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.input_fc = nn.Linear(input_dim, output_dim)
        self.input_fc = nn.Linear(input_dim, 100)
        self.hidden_fc = nn.Linear(100, 50)
        self.output_fc = nn.Linear(50, output_dim)
        
    def forward(self, x):
        #x = [batch size, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1).float()
        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        # y_pred = self.input_fc(x)
        
        return y_pred
    

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
    num_labels = len(label_list)
    
    epoch = args.num_train_epochs
    bs = args.train_batch_size
    
    # Tokenize the text
    character_tokenizer = BertTokenizer.from_pretrained('./pretrained-models/bert-base-uncased/')
    indexer = CharacterIndexer()
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    character_model = CharacterBertModel.from_pretrained('./pretrained-models/general_character_bert/').to('cuda')
    bert_model = AutoModel.from_pretrained('bert-base-uncased').to('cuda')

    mlp_model = MLP(1536, 2).to('cuda')
    criterion = CrossEntropyLoss()
    if args.fp16:
        bert_model.half()
        character_model.half()
        mlp_model.half()
    
    alpha=torch.nn.parameter.Parameter(torch.tensor(1.0).to('cuda'))
    alpha.requires_grad=True
    param_optimizer = list(bert_model.named_parameters())
    mlp_optimizer =  list(mlp_model.named_parameters())
    character_optimizer = list(character_model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in mlp_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in mlp_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {"params": [p for n, p in character_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in character_optimizer if any(nd in n for nd in no_decay)],"weight_decay": 0.0},
        {'params': [alpha], 'weight_decay': 0.01}, 
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
                              eps=1e-8,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          eps=1e-8)

    # for BERT
    train_examples, train_golden_labels, seen_types = processor.get_examples_title_train('./data/classification_new/train.txt', 100000)
    train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, bert_tokenizer, output_mode)
            
    all_train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_labels = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    
    test_examples, test_golden_labels, seen_types = processor.get_examples_title_test('./data/classification_new/test.txt', 100000)
    test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, bert_tokenizer, output_mode)
            
    all_test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_labels = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    all_test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    
    # for CharacterBERT
    data = {}
    global_labels = {}
    func = load_classification_dataset
    tokenization_function = character_tokenizer.tokenize
    for split in ['train', 'test']:
        data[split], global_labels[split] = func(step=split, do_lower_case=True)
        retokenize(data[split], tokenization_function)

    # Count target labels or classes
    counter_all = Counter(
        [example.label for example in data['train'] + data['test']])
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
            examples=data[split],
            labels=labels,
            pad_token_id=pad_token_id,
            pad_token_label_id=pad_token_label_id,
            max_seq_length=max_seq_length)

    del data
    
    train_data = TensorDataset(all_train_input_ids, all_train_input_mask, all_train_segment_ids, train_labels, dataset['train'][0], dataset['train'][1], dataset['train'][2], dataset['train'][3])
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)
    
    test_data = TensorDataset(all_test_input_ids, all_test_input_mask, all_test_segment_ids, test_labels, dataset['test'][0], dataset['test'][1], dataset['test'][2], dataset['test'][3])
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=bs)
    
    best_valid_loss = float('inf')
    for _ in trange(int(epoch), desc="Epoch"):
        pred_probs = []
        for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration", position=0, leave=True)):
            bert_model.train()
            character_model.train()
            mlp_model.train()
            
            batch = tuple(t.to('cuda') for t in batch)
            input_ids, input_mask, segment_ids, label_ids, character_input_ids, character_input_mask, character_segment_ids, character_label_ids = batch
            
            bert_embedding = bert_model(input_ids, input_mask)[0][:, 0, :]
            
            embeddings_for_batch, _ = character_model(character_input_ids, character_input_mask)
            character_embedding = embeddings_for_batch[:, 0, :]
            
            embedding = torch.cat((bert_embedding, alpha*character_embedding), 1)
            # embedding = bert_embedding+alpha*character_embedding
            
            logits = mlp_model(embedding)
            loss_fct = CrossEntropyLoss()
            train_loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))
            
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(alpha)
    
        test_tr_loss = 0
        pred_probs = []
        for input_ids, input_mask, segment_ids, label_ids, character_input_ids, character_input_mask, character_segment_ids, character_label_ids in test_dataloader:
            bert_model.eval()
            character_model.eval()
            mlp_model.eval()
            
            input_ids = input_ids.to('cuda')
            input_mask = input_mask.to('cuda')
            segment_ids = segment_ids.to('cuda')
            label_ids = label_ids.to('cuda') 
            
            character_input_ids = character_input_ids.to('cuda')
            character_input_mask = character_input_mask.to('cuda')
            character_segment_ids = character_segment_ids.to('cuda')
            character_label_ids = character_label_ids.to('cuda')
            
            with torch.no_grad():
                bert_embedding = bert_model(input_ids, input_mask)[0][:, 0, :]
                
                embeddings_for_batch, _ = character_model(character_input_ids, character_input_mask)
                character_embedding = embeddings_for_batch[:, 0, :]
                
                embedding = torch.cat((bert_embedding, alpha*character_embedding), 1)
                # embedding = bert_embedding+alpha*character_embedding
                
                logits = mlp_model(embedding)
                
                loss_fct = CrossEntropyLoss()
                test_loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))
                
                if len(pred_probs) == 0:
                    pred_probs.append(logits.detach().cpu().numpy())
                else:
                    pred_probs[0] = np.append(pred_probs[0], logits.detach().cpu().numpy(), axis=0)
                
            test_tr_loss += test_loss
            
                
        if test_tr_loss < best_valid_loss:
            best_valid_loss = test_tr_loss
            torch.save(mlp_model.state_dict(), 'tut1-model.pt')

        print(f'\tTrain Loss: {train_loss:.8f}')
        print(f'\t Val. Loss: {test_tr_loss:.8f}')
        
        dic = dict()
        dic[0] = 'brandname'
        dic[1] = 'product'
        dic[2] = 'features'
        test_preds = []
        pred_probs=pred_probs[0]
        pred_probs = softmax(pred_probs,axis=1)
        pred_probs = list(pred_probs[:,0])
        for i in range(0, len(pred_probs), 3):
            my_list = [pred_probs[i], pred_probs[i+1], pred_probs[i+2]]
            # print(my_list)
            my_max = max(my_list)
            my_index = my_list.index(my_max)
            test_preds.append(my_index)
        print(test_preds[:10])
        print(test_golden_labels[:10])
        print(classification_report(test_golden_labels, test_preds, digits=4))
        print(confusion_matrix(test_golden_labels, test_preds))


    

if __name__ == "__main__":
    main()
    
    
