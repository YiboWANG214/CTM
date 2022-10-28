"""
Fine-tune bioBERT on our dataset.
"""

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForMaskedLM
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
import time
import datetime
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import RobertaConfig, RobertaModel

if torch.cuda.is_available():    
   
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


lr = 0.00002

# Roberta:
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
# pretrained_model = BertForMaskedLM.from_pretrained("roberta-large")
pretrained_model = RobertaForMaskedLM.from_pretrained(
                                                        "roberta-large",
                                                        output_attentions = False, # Whether the model returns attentions 
                                                        output_hidden_states = True, # Whether the model returns all hidden-states.
                                                        )

pretrained_model.cuda()
# pretrained_model.eval()

optimizer = AdamW(pretrained_model.parameters(), lr=lr, eps = 1e-8)

data = []
f = open('corpus.txt', 'r')
f = f.readlines()
# for line in f:
#     data.append(line)
print(len(f))
for i in range(100000):
    line = f[i]
    data.append(line)

print("data ready")

input_ids = []
attention_masks = []
labels = []
inputs = []
for sentence in data:
    # encoded_sentence = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    encoded_sentence = tokenizer.encode_plus(
                        sentence, 
                        add_special_tokens = True,
                        max_length = 32,
                        truncation=True,
                        pad_to_max_length = True,
                        # padding = 'max_length',
                        return_attention_mask = True, 
                        return_tensors = 'pt',
                   )
    input_ids.append(encoded_sentence['input_ids'])
    attention_masks.append(encoded_sentence['attention_mask'])
    # inputs.append(encoded_sentence)
    # labels.append(encoded_sentence['input_ids'])

print("embeddings ready")

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

dataset = TensorDataset(input_ids, attention_masks)
# dataset = TensorDataset(inputs, labels)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))


batch_size = 16

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


epochs = 1

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


training_stats = []
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    train_loss = 0
    pretrained_model.train()

    for step, batch in enumerate(train_dataloader):

        if step % 320 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        pretrained_model.zero_grad()
        pretrained_outputs = pretrained_model(batch[0].to(device), 
                                              attention_mask=batch[1].to(device),
                                              masked_lm_labels=batch[0].to(device))
        # pretrained_outputs = pretrained_model(**batch[0].to(device), labels=batch[1].to(device))
        pretrained_loss = pretrained_outputs[0]
        train_loss += pretrained_loss.item()
        pretrained_loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = train_loss / len(train_dataset)
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.8f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    print("")
    print("Running Validation...")

    t0 = time.time()

    eval_loss = 0
    pretrained_model.eval()

    # Evaluate data for one epoch
    
    for batch in validation_dataloader:
        with torch.no_grad():
            outputs = pretrained_model(batch[0].to(device), 
                                      attention_mask=batch[1].to(device),
                                      masked_lm_labels=batch[0].to(device))        
            # outputs = pretrained_model(**batch[0].to(device), labels=batch[1].to(device))

            loss = outputs[0]

        eval_loss += loss.item()

    avg_val_loss = eval_loss / len(val_dataset)
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.8f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            # 'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")


# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = './model_save_title/'

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