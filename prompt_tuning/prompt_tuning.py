import os
from pathlib import Path

from transformers import BertLMHeadModel, BertModel, BertTokenizer
import torch
import torch.nn as nn



bert_model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  ).to('cuda')
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model.eval()


class BertPromptTuningMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        soft_prompt_path: str = None,
        n_tokens: int = None,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        elif n_tokens is not None:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )

        return model

    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path
        """
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device("cpu")
        )
        self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def initialize_soft_prompt(
        self,
        n_tokens: int = 1,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens

        token_label = bert_tokenizer.tokenize('is a band name, which is a type of things manufactured by a particular company under a particular name')
        input_ids_label = bert_tokenizer.convert_tokens_to_ids(token_label)
        input_ids_label = torch.tensor([input_ids_label], dtype=torch.long).to('cuda')
        init_prompt_value = bert_model(input_ids_label)[0].view(self.n_tokens, 768)
        
        # # random
        self.soft_prompt = nn.Embedding(n_tokens, 768)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = bert_model(input_ids)[0]

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)
        new_inputs_embeds = []
        for i in range(inputs_embeds.shape[0]):
            if 0 in input_ids[i].tolist():
                tmp_index = input_ids[i].tolist().index(0)
                tmp = torch.cat([inputs_embeds[i][:tmp_index], learned_embeds[i], inputs_embeds[i][tmp_index:]], dim=0)
            else:
                tmp = torch.cat([inputs_embeds[i], learned_embeds[i]], dim=0)
            new_inputs_embeds.append(tmp)
        new_inputs_embeds = torch.stack(new_inputs_embeds)
        
        return new_inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)
        n_batches = labels.shape[0]

        learned_labels = torch.full((n_batches, self.n_tokens), ignore_index).to(self.device)
        new_labels = []
        for i in range(n_batches):
            if 0 in labels[i]:
                tmp_index = labels[i].tolist().index(0)
                tmp = torch.cat([labels[i][:tmp_index], learned_labels[i], labels[i][tmp_index:]], dim=0)
            else:
                tmp = torch.cat([labels[i], learned_labels[i]], dim=0)
            new_labels.append(tmp)
        new_labels = torch.stack(new_labels)
        return new_labels
        

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)
        n_batches = attention_mask.shape[0]
        
        learned_mask = torch.full((n_batches, self.n_tokens), 1).to(self.device)
        new_mask = []
        for i in range(n_batches):
            print(attention_mask[i])
            if 0 in attention_mask[i]:
                tmp_index = attention_mask[i].tolist().index(0)
                tmp = torch.cat([attention_mask[i][:tmp_index], learned_mask[i], attention_mask[i][tmp_index:]], dim=0)
            else:
                tmp = torch.cat([attention_mask[i], learned_mask[i]], dim=0)
            new_mask.append(tmp)
        new_mask = torch.stack(new_mask)
        return new_mask
        
    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
                self.device
            )

        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )


class BertPromptTuningLM(BertPromptTuningMixin, BertLMHeadModel):
    def __init__(self, config):
        super().__init__(config)


