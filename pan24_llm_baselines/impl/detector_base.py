from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, \
    PreTrainedModel, PreTrainedTokenizerBase


class DetectorBase(ABC):
    """
    LLM detector base class.

    Implements shared functionality useful for implementing transformer-based LLM detectors.
    """

    @abstractmethod
    def get_score(self, text: Union[str, List[str]]) -> npt.NDArray[np.float64]:
        """
        Return a prediction score indicating the "humanness" of the input text.

        :param text: input text or list of input texts
        :return: humanness score(s)
        """

    @abstractmethod
    def predict(self, text: Union[str, List[str]]) -> npt.NDArray[np.bool_]:
        """
        Make a prediction whether the input text was written by a human.

        :param text: input text or list of input texts
        :return: boolean values indicating whether inputs are classified as human
        """

    @staticmethod
    @torch.inference_mode()
    def _load_model(model_name: str,
                    device_map: Union[str, Dict[str, Union[int, str, torch.device]], int, torch.device],
                    auto_cls: type = AutoModelForCausalLM,
                    use_flash_attn=False,
                    quantization_bits=None,
                    trust_remote_code=False,
                    torch_dtype: torch.dtype = torch.bfloat16,
                    **additional_args) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:

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
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    @staticmethod
    @torch.inference_mode()
    def _tokenize(batch: Union[str, List[str]],
                  tokenizer: PreTrainedTokenizerBase,
                  device: torch.device = None,
                  max_length: int = None,
                  **additional_args) -> transformers.BatchEncoding:
        batch = [batch] if isinstance(batch, str) else batch
        encodings = tokenizer(
            batch,
            return_tensors='pt',
            padding='longest' if len(batch) > 1 else False,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,
            **additional_args
        )
        if device:
            return encodings.to(device)
        return encodings

    @staticmethod
    @torch.inference_mode()
    def _perplexity(logits_or_model: Union[torch.Tensor, PreTrainedModel],
                    encoding: transformers.BatchEncoding) -> npt.NDArray[np.float64]:
        """
        Calculate model perplexity / negative log loss on a batch of input sequences.

        :param logits_or_model: predicted logits (will be shifted to match input) or causal LM model
        :param encoding: input encoding
        :return: model perplexity on the sequence
        """

        if isinstance(logits_or_model, PreTrainedModel):
            if encoding.input_ids.shape[0] == 1:
                # Batch size == 1
                return logits_or_model(**encoding, labels=encoding.input_ids).loss
            logits_or_model = logits_or_model(**encoding).logits

        logits_or_model = logits_or_model[..., :-1, :]
        labels = encoding.input_ids[..., 1:]
        attention_mask = encoding.attention_mask[..., 1:]

        ppl = F.cross_entropy(logits_or_model.transpose(1, 2), labels, reduction='none')
        ppl = (ppl * attention_mask).sum(1) / attention_mask.sum(1)
        return ppl.cpu().float().numpy()

    @staticmethod
    @torch.inference_mode()
    def _entropy(p_logits: torch.Tensor, discard_last: bool = True) -> npt.NDArray[np.float64]:
        """
        Calculate entropy of predicted logits.

        :param p_logits: predicted logits
        :param discard_last: discard last logit
        :return: logit softmax entropy
        """

        if discard_last:
            p_logits = p_logits[..., :-1, :]
        e = -(F.softmax(p_logits, -1) * F.log_softmax(p_logits, -1)).sum(-1).mean()
        return e.cpu().float().numpy()

    @staticmethod
    @torch.inference_mode()
    def _cross_entropy(p_logits: torch.Tensor, q_logits: torch.Tensor, mask: torch.tensor) -> npt.NDArray[np.float64]:
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
        ce = (ce * mask).sum(1) / mask.sum(1)
        return ce.cpu().float().numpy()
