"""
Prompt tuning model
reference: https://github.com/mkshing/Prompt-Tuning/blob/master/model.py
"""
import os
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

from transformers import BertLMHeadModel, BertModel, BertTokenizer, BertPreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.activations import ACT2FN
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from modeling.character_bert import CharacterBertModel
from utils.character_cnn import CharacterIndexer
from modeling.character_cnn import CharacterCNN

bert_model = CharacterBertModel.from_pretrained('./pretrained-models/general_character_bert/',
                                  output_hidden_states = True,).to('cuda')
bert_tokenizer = BertTokenizer.from_pretrained('./pretrained-models/bert-base-uncased/')
# indexer = CharacterIndexer()
bert_tokenizer = bert_tokenizer.basic_tokenizer
characters_indexer = CharacterIndexer()
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
        # from label
        # long 18
        # brand: a band name, which is a type of things manufactured by a particular company under a particular name
        # brand: brand name brand name brand name brand name brand name brand name brand name brand name brand name 
        # brand: brand / trademark
        # product: a product, which is an article or substance that is manufactured or refined for sale. product product
        # product: product product product product product product product product product product product product product product product product product product
        # product: product / commodity
        # features: a feature, which is a distinctive attribute or aspect of something. feature feature feature feature feature feature
        # features: feature feature feature feature feature feature feature feature feature feature feature feature feature feature feature feature feature feature
        # features: feature / aspect
        token_label = bert_tokenizer.tokenize('is a brand name, which is a type of things manufactured by a particular company under a particular name')
        input_ids_label = characters_indexer.as_padded_tensor([token_label])[0].view(1, -1, 50).to('cuda')
        init_prompt_value = bert_model(input_ids_label)[0].view(-1, 768)
        # print(init_prompt_value.shape)
        
        # # random
        self.soft_prompt = CharacterCNN(
            requires_grad=True,
            output_dim=768)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        # inputs_embeds = self.transformer.wte(input_ids)
        inputs_embeds = bert_model(input_ids)[0]

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)
        new_inputs_embeds = []
        for i in range(inputs_embeds.shape[0]):
            # print(input_ids[i].tolist())
            if [0]*50 in input_ids[i].tolist():
                # print(i)
                tmp_index = input_ids[i].tolist().index([0]*50)
                tmp = torch.cat([inputs_embeds[i][:tmp_index], learned_embeds[i], inputs_embeds[i][tmp_index:]], dim=0)
            else:
                # print(i)
                tmp = torch.cat([inputs_embeds[i], learned_embeds[i]], dim=0)
            new_inputs_embeds.append(tmp)
        new_inputs_embeds = torch.stack(new_inputs_embeds)
        # inputs_embeds = torch.cat([inputs_embeds, learned_embeds], dim=1)

        # print('inputs_embeds', new_inputs_embeds.shape)
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

        
        # return torch.cat(
        #     [
        #         labels,
        #         torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
        #     ],
        #     dim=1,
        # )

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
        
        # return torch.cat(
        #     [attention_mask, torch.full((n_batches, self.n_tokens), 1).to(self.device)],
        #     dim=1,
        # )

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
            # print(labels)
        # labels = labels[:, :, 0]

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)
        
        # print('===++++====+++++=====')
        # print('inputs_embeds', inputs_embeds.shape)
        # Drop most of the args for now
        result = super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        return result


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # print(hidden_states.shape)
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states



class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores



class CharacterBertLMHeadModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.bert = CharacterBertModel(config)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
            encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be
                in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100`
                are ignored (masked), the loss is only computed for the tokens with labels n `[0, ...,
                config.vocab_size]`
            past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up
                decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        Returns:
        Example:
        ```python
        >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
        >>> import torch
        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        >>> config = BertConfig.from_pretrained("bert-base-cased")
        >>> config.is_decoder = True
        >>> model = BertLMHeadModel.from_pretrained("bert-base-cased", config=config)
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> prediction_logits = outputs.logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            # past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
            # cross_attentions=outputs.cross_attentions,
        )
        

        
class BertPromptTuningLM(BertPromptTuningMixin, CharacterBertLMHeadModel):
    def __init__(self, config):
        super().__init__(config)