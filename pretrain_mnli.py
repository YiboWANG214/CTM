"""
Fine-tune bioBERT on our dataset.
"""

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForMaskedLM, AutoModelForSequenceClassification
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss, MSELoss
import os
import time
import datetime
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import RobertaConfig, RobertaModel
import json
from sklearn.metrics import classification_report
import ast
import numpy as np
from scipy.special import softmax

if torch.cuda.is_available():    
   
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

output_dir = './model_save_title/'
lr = 0.00002
max_seq_length = 128

pretrained_model = AutoModelForSequenceClassification.from_pretrained(output_dir,
                                        output_attentions=False,  # Whether the model returns attentions weights.
                                        output_hidden_states=True,  # Whether the model returns all hidden-states.
                                        num_labels=3,
                                        )
tokenizer = AutoTokenizer.from_pretrained(output_dir)

pretrained_model.cuda()
pretrained_model.eval()

optimizer = AdamW(pretrained_model.parameters(), lr=lr, eps = 1e-8)


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

file = open('multinli_1.0_train.jsonl', 'r', encoding='utf-8')
f = file.readlines()
print(len(f))
label_map = {"entailment":0, "contradiction":1, "neutral":2}
input_ids = []
attention_masks = []
segment_ids = []
label_ids = []

premise_2_tokenzed = {}
hypothesis_2_tokenzed = {}
list_2_tokenizedID = {}
for i in range(len(f)):
    # print(i)
    sample = f[i]
    sample = ast.literal_eval(sample)
    
    tokens_a = premise_2_tokenzed.get(sample["sentence1"])
    if tokens_a is None:
        tokens_a = tokenizer.tokenize(sample["sentence1"])
        premise_2_tokenzed[sample["sentence1"]] = tokens_a
    tokens_b = premise_2_tokenzed.get(sample["sentence2"])
    if tokens_b is None:
        tokens_b = tokenizer.tokenize(sample["sentence2"])
        hypothesis_2_tokenzed[sample["sentence2"]] = tokens_b
        
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens_A = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids_A = [0] * len(tokens_A)
    tokens_B = tokens_b + ["[SEP]"]
    segment_ids_B = [1] * (len(tokens_b) + 1)
    tokens = tokens_A + tokens_B
    tmp_segment_ids = segment_ids_A + segment_ids_B
    
    input_ids_A = list_2_tokenizedID.get(' '.join(tokens_A))
    if input_ids_A is None:
        input_ids_A = tokenizer.convert_tokens_to_ids(tokens_A)
        list_2_tokenizedID[' '.join(tokens_A)] = input_ids_A
    input_ids_B = list_2_tokenizedID.get(' '.join(tokens_B))
    if input_ids_B is None:
        input_ids_B = tokenizer.convert_tokens_to_ids(tokens_B)
        list_2_tokenizedID[' '.join(tokens_B)] = input_ids_B 
    tmp_input_ids = input_ids_A + input_ids_B

    tmp_attention_mask = [1] * len(tmp_input_ids)
    
    padding = [0] * (max_seq_length - len(tmp_input_ids))
    tmp_input_ids += padding
    tmp_attention_mask += padding
    tmp_segment_ids += padding
    
    # print(len(tmp_input_ids))
    assert len(tmp_input_ids) == max_seq_length
    assert len(tmp_attention_mask) == max_seq_length
    assert len(tmp_segment_ids) == max_seq_length
    
    attention_masks.append(tmp_attention_mask)
    segment_ids.append(tmp_segment_ids)
    input_ids.append(tmp_input_ids)
    
    tmp_label_id = label_map[sample["annotator_labels"][0]]
    label_ids.append(tmp_label_id)

print("embeddings ready")

# input_ids = torch.cat(input_ids, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)
# segment_ids = torch.cat(segment_ids, dim=0)
# label_ids = torch.cat(label_ids, dim=0)

input_ids = torch.tensor([tmp_input_ids for tmp_input_ids in input_ids], dtype=torch.long)
attention_masks = torch.tensor([tmp_attention_masks for tmp_attention_masks in attention_masks], dtype=torch.long)
segment_ids = torch.tensor([tmp_segment_ids for tmp_segment_ids in segment_ids], dtype=torch.long)
label_ids = torch.tensor([tmp_label_id for tmp_label_id in label_ids], dtype=torch.long)

train_size = int(0.9 * len(label_ids))
val_size = len(label_ids) - train_size

train_input_ids, train_attention_masks, train_label_ids = input_ids[:train_size], attention_masks[:train_size], label_ids[:train_size]
val_input_ids, val_attention_masks, val_label_ids = input_ids[train_size:], attention_masks[train_size:], label_ids[train_size:]
print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_label_ids)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_label_ids)

batch_size = 8

train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
        )


epochs = 3

total_steps = len(train_dataset) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

global_step = 0
training_stats = []
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

        if step % 320 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        pretrained_model.train()
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch
        logits = pretrained_model(input_ids, input_mask)[0]
        # print(logits)
        # print(len(logits))
        loss_fct = CrossEntropyLoss()
        # print(logits)
        # print(logits[0].shape)
        # print(logits[1][0].shape)
        loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))
        loss = loss.mean()

        loss.backward()

        logits = logits[0]
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
    
    print("===========start inference============")
    pretrained_model.eval()

    val_tr_loss = 0
    nb_val_steps = 0
    preds = []
    print('Inferencing...')
    for input_ids, input_mask, label_ids in validation_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = pretrained_model(input_ids, input_mask)[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))
            loss = loss.mean()
        val_tr_loss += loss.item()
        
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = softmax(preds[0],axis=1)
    pred_labels = []
    for tmp in preds:
        tmp = list(tmp)
        tmp_label = tmp.index(max(tmp))
        pred_labels.append(tmp_label)
    print(classification_report(val_label_ids, pred_labels))
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': tr_loss / len(train_dataset),
            'Valid. Loss': val_tr_loss / len(val_dataset),
        }
    )

print("")
print("Training complete!")
print(training_stats)


# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = './model_save_title_mnli/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = pretrained_model.module if hasattr(pretrained_model, 'module') else pretrained_model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))