# Copyright 2024 Janek Bevendorff, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Type, Union

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


AutoModelClsType = Type[transformers.models.auto.auto_factory._BaseAutoModelClass]
TorchDeviceMapType = Union[str, Dict[str, Union[int, str, torch.device]], int, torch.device]


@torch.inference_mode()
def load_model(model_name: str,
               device_map: TorchDeviceMapType,
               auto_cls: AutoModelClsType = AutoModelForCausalLM,
               use_flash_attn=False,
               quantization_bits=None,
               trust_remote_code=False,
               torch_dtype: torch.dtype = torch.bfloat16,
               **additional_args) -> transformers.PreTrainedModel:

    model_args = {
        'trust_remote_code': trust_remote_code,
        'torch_dtype': torch_dtype,
        **additional_args
    }

    if use_flash_attn:
        model_args.update({'attn_implementation': 'flash_attention_2'})

    if quantization_bits:
        model_args.update({
            'quantization_config': BitsAndBytesConfig(**{
                f'load_in_{quantization_bits}bit': True,
                f'bnb_{quantization_bits}bit_compute_dtype': torch.bfloat16
            })
        })

    model = auto_cls.from_pretrained(model_name, device_map=device_map, **model_args)
    model.eval()
    return model


@torch.inference_mode()
def load_tokenizer(model_name: str, **tokenizer_args) -> transformers.PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@torch.inference_mode()
def tokenize_sequences(batch: Union[str, List[str]],
                       tokenizer: transformers.PreTrainedTokenizerBase,
                       device: Union[str, torch.device] = None,
                       max_length: int = None,
                       return_tensors='pt',
                       **additional_args) -> transformers.BatchEncoding:
    batch = [batch] if isinstance(batch, str) else batch
    args = dict(
        return_tensors=return_tensors,
        padding='longest' if len(batch) > 1 else False,
        truncation=max_length is not None,
        max_length=max_length,
        return_token_type_ids=False,
    )
    args.update(additional_args)
    encodings = tokenizer(batch, **args)
    if device and return_tensors == 'pt':
        return encodings.to(device)
    return encodings


@torch.inference_mode()
def log_likelihood(logits_or_model: Union[torch.Tensor, transformers.PreTrainedModel],
                   encoding: transformers.BatchEncoding, batch_size: Optional[int] = None) -> torch.Tensor:
    """
    Calculate negative log loss / model log perplexity on a batch of input sequences.

    :param logits_or_model: predicted logits (will be shifted to match input) or causal LM model
    :param encoding: input encoding
    :param batch_size: evaluation batch size
    :return: summed log likelihood of the sequence according to the model
    """

    if isinstance(logits_or_model, transformers.PreTrainedModel):
        if encoding.input_ids.shape[0] == 1:
            # Batch size == 1
            return logits_or_model(**encoding, labels=encoding.input_ids).loss

        if batch_size is None:
            batch_size = len(encoding.input_ids)
        m = logits_or_model
        logits_or_model = []
        for b in range(0, len(encoding.input_ids), batch_size):
            logits_or_model.append(m(**{k: v[b:b + batch_size] for k, v in encoding.items()}).logits)
        logits_or_model = torch.vstack(logits_or_model)

    logits_or_model = logits_or_model[..., :-1, :]
    labels = encoding.input_ids[..., 1:]
    attention_mask = encoding.attention_mask[..., 1:]

    ll = F.cross_entropy(logits_or_model.transpose(1, 2), labels, reduction='none')
    return (ll * attention_mask).sum(1) / attention_mask.sum(1)


@torch.inference_mode()
def entropy(p_logits: torch.Tensor, discard_last: bool = True) -> torch.Tensor:
    """
    Calculate entropy of predicted logits.

    :param p_logits: predicted logits
    :param discard_last: discard last logit
    :return: logit softmax entropy
    """

    if discard_last:
        p_logits = p_logits[..., :-1, :]
    return -(F.softmax(p_logits, -1) * F.log_softmax(p_logits, -1)).sum(-1).mean()


@torch.inference_mode()
def cross_entropy(p_logits: torch.Tensor, q_logits: torch.Tensor, mask: torch.tensor) -> torch.Tensor:
    """
    Calculate cross entropy between two distributions of predicted logits.

    :param p_logits: true logits
    :param q_logits: predicted q logits
    :param mask: padding mask
    :return: logit softmax cross entropy
    """
    vocab_size = p_logits.shape[-1]
    n_tokens = p_logits.shape[-2]

    p_prob = F.softmax(p_logits, -1).view(-1, vocab_size)
    q_logits = q_logits.view(-1, vocab_size)

    ce = F.cross_entropy(input=q_logits, target=p_prob, reduction='none').view(-1, n_tokens)
    return (ce * mask).sum(1) / mask.sum(1)
