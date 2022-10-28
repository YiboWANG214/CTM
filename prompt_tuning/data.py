""" Tools for loading datasets as Classification/SequenceLabelling Examples. """
import os
import logging
from collections import namedtuple

from tqdm import tqdm
import random
from transformers import BasicTokenizer

from utils.data import retokenize

from collections import defaultdict, OrderedDict


DATA_PATH = 'data/'
ClassificationExample = namedtuple(
    'ClassificationExample', ['id', 'tokens_a', 'tokens_b', 'label'])
SequenceLabellingExample = namedtuple(
    'SequenceLabellingExample', ['id', 'token_sequence', 'label_sequence'])



def load_classification_dataset(step, do_lower_case, category):
    """ Loads classification exampels from a dataset. """
    assert step in ['train', 'test']
    basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    path = DATA_PATH + f'{step}' + '_new.txt'
    examples = []
    global_labels = []
    
    dic = dict()
    dic['brandname'] = 0
    dic['product'] = 1
    dic['features'] = 2
    with open(path, 'r', encoding='utf-8') as data_file:
        lines = data_file.readlines()
        history_types = set(['brandname', 'product', 'features'])
        for i, line in tqdm(enumerate(lines), desc=f'reading `{os.path.basename(path)}`...'):
            # example: product dress  Spring and summer new sexy V-neck ruffled lace-up dress, chiffon beach skirt, wrap dress CN-9
            category_token = line.split('  ')[0]
            type_list = [category_token.split(' ')[0]]
            token = ' '.join(category_token.split(' ')[1:])
            text = ' '.join(line.split('  ')[1:])
            
            neg_types = history_types - set(type_list)

            if len(neg_types) > 1:
                sampled_type_set = random.sample(neg_types, 1)
            else:
                continue

            '''pos pair'''
            text_a = text
            
            if type_list[0] == category:
                # print(type_list[0], token)
                guid = "train-" + str(i)
                text_b = token
                label = category
                global_labels.append(type_list[0])
                # global_labels.append(dic[type_list[0]])
                examples.append(
                    ClassificationExample(
                        id=i,
                        tokens_a=text_a,
                        tokens_b=text_b,
                        label=label,
                    )
                )
    
    logging.info('Number of `%s` examples: %d', step, len(examples))
    return examples, global_labels
    
