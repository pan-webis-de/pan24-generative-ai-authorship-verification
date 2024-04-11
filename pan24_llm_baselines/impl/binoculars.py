# Refactored version of Hans et al.'s Binoculars LLM detector
#
# Copyright (c) 2023, Abhimanyu Hans, Avi Schwarzschild, Tom Goldstein
# Copyright (c) 2024, Janek Bevendorff
#
# Hans, Abhimanyu, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi,
# Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024.
# “Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated
# Text.” arXiv [Cs.CL]. arXiv. http://arxiv.org/abs/2401.12070.
#
# Original GitHub: https://github.com/ahans30/Binoculars/tree/main


from typing import Dict, Literal, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import transformers

from pan24_llm_baselines.impl.detector_base import DetectorBase


class Binoculars(DetectorBase):
    # Selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
    BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
    BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

    def __init__(self,
                 observer_name_or_path='tiiuae/falcon-7b',
                 performer_name_or_path='tiiuae/falcon-7b-instruct',
                 device1: Union[str, Dict[str, Union[int, str, torch.device]], int, torch.device] = 'auto',
                 device2: Union[str, Dict[str, Union[int, str, torch.device]], int, torch.device] = 'auto',
                 mode: Literal['low-fpr', 'accuracy'] = 'low-fpr',
                 max_token_observed=512,
                 use_flash_attn=False,
                 quantization_bits=None,
                 **model_args):

        self.threshold = None
        self.change_mode(mode)

        self.observer_model, self.tokenizer = self._load_model(
            observer_name_or_path,
            device_map=device1,
            use_flash_attn=use_flash_attn,
            quantization_bits=quantization_bits,
            **model_args)

        self.performer_model, perf_tokenizer = self._load_model(
            performer_name_or_path,
            device_map=device2,
            use_flash_attn=use_flash_attn,
            quantization_bits=quantization_bits,
            **model_args)

        if not hasattr(self.tokenizer, 'vocab') or self.tokenizer.vocab != perf_tokenizer.vocab:
            raise ValueError(f'Incompatible tokenizers for {observer_name_or_path} and {performer_name_or_path}.')

        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == 'low-fpr':
            self.threshold = self.BINOCULARS_FPR_THRESHOLD
        elif mode == 'accuracy':
            self.threshold = self.BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f'Invalid mode: {mode}')

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> Tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(**encodings.to(self.observer_model.device)).logits
        performer_logits = self.performer_model(**encodings.to(self.performer_model.device)).logits

        if next(self.observer_model.parameters()).is_cuda:
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    @torch.inference_mode()
    def get_score(self, text: Union[str, List[str]]) -> npt.NDArray[np.float64]:
        encodings = self._tokenize(text, self.tokenizer, self.observer_model.device)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = self._perplexity(performer_logits, encodings)
        x_ppl = self._cross_entropy(observer_logits,
                                    performer_logits.to(self.observer_model.device),
                                    encodings.attention_mask)
        binoculars_scores = ppl / x_ppl
        return binoculars_scores[0] if isinstance(text, str) else binoculars_scores

    def predict(self, text: Union[str, List[str]]) -> npt.NDArray[np.bool_]:
        return self.get_score(text) > self.threshold
