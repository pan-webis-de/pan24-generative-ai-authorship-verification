# Slightly modified version of Hans et al.'s Binoculars LLM detector
#
# Copyright (c) 2023, Abhimanyu Hans, Avi Schwarzschild, Tom Goldstein
# https://github.com/ahans30/Binoculars/tree/main

from typing import Union

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pan24_llm_baselines.thirdparty_binoculars.utils import assert_tokenizer_consistency
from pan24_llm_baselines.thirdparty_binoculars.metrics import perplexity, entropy

torch.set_grad_enabled(False)

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]


class Binoculars(object):
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 device1='auto',
                 device2='auto',
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 use_flash_attn=False,
                 quantization_bits=None,
                 trust_remote_code=False
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        model_args = {
            'trust_remote_code': trust_remote_code,
            'torch_dtype': torch.bfloat16 if use_bfloat16 else torch.float32
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

        self.change_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   device_map=device1,
                                                                   **model_args)
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    device_map=device2,
                                                                    **model_args)
        self.observer_model.eval()
        self.performer_model.eval()

        self.device1 = self.observer_model.device
        self.device2 = self.performer_model.device

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(self.device1)).logits
        performer_logits = self.performer_model(**encodings.to(self.device2)).logits
        if self.device1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(self.device1), performer_logits.to(self.device1),
                        encodings.to(self.device1), self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred
