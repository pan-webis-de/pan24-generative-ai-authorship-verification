# Refactored version of Hans et al.'s Binoculars LLM detector
#
# BSD 3-Clause License
#
# Copyright (c) 2023, Abhimanyu Hans, Avi Schwarzschild, Tom Goldstein
# Copyright 2024 Janek Bevendorff, Webis
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import List, Literal, Tuple

import torch
import transformers

from pan24_llm_baselines.detectors.detector_base import DetectorBase
from pan24_llm_baselines.util import *

__all__ = ['Binoculars']


class Binoculars(DetectorBase):
    """
    Binoculars LLM detector.

    This is a refactored implementation of the original: https://github.com/ahans30/Binoculars/tree/main

    References:
    ===========
        Hans, Abhimanyu, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi,
        Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024.
        “Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated
        Text.” arXiv [Cs.CL]. arXiv. http://arxiv.org/abs/2401.12070.
    """
    # Selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
    BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
    BINOCULARS_FPR_THRESHOLD = 0.8536432310785527       # optimized for low-fpr [chosen at 0.01%]

    def __init__(self,
                 mode: Literal['low-fpr', 'accuracy'] = 'low-fpr',
                 observer_name_or_path='tiiuae/falcon-7b',
                 performer_name_or_path='tiiuae/falcon-7b-instruct',
                 device1: TorchDeviceMapType = 'auto',
                 device2: TorchDeviceMapType = 'auto',
                 max_token_observed=512,
                 use_flash_attn=False,
                 quantization_bits=None,
                 **model_args):
        """
        :param mode: prediction mode
        :param observer_name_or_path: observer model
        :param performer_name_or_path: performer model
        :param device1: observer device
        :param device2: performer device
        :param max_token_observed: max number of tokens to analyze
        :param use_flash_attn: use flash attention
        :param quantization_bits: quantize model
        :param model_args: additional model args
        """

        self.scoring_mode = mode
        self.observer_model = load_model(
            observer_name_or_path,
            device_map=device1,
            use_flash_attn=use_flash_attn,
            quantization_bits=quantization_bits,
            **model_args)
        self.tokenizer = load_tokenizer(observer_name_or_path)

        self.performer_model = load_model(
            performer_name_or_path,
            device_map=device2,
            use_flash_attn=use_flash_attn,
            quantization_bits=quantization_bits,
            **model_args)
        perf_tokenizer = load_tokenizer(performer_name_or_path)

        if not hasattr(self.tokenizer, 'vocab') or self.tokenizer.vocab != perf_tokenizer.vocab:
            raise ValueError(f'Incompatible tokenizers for {observer_name_or_path} and {performer_name_or_path}.')

        self.max_token_observed = max_token_observed

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> Tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(**encodings.to(self.observer_model.device)).logits
        performer_logits = self.performer_model(**encodings.to(self.performer_model.device)).logits

        if next(self.observer_model.parameters()).is_cuda:
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def _normalize_scores(self, scores):
        return torch.sigmoid(-10 * self.threshold * (scores - self.threshold))

    @torch.inference_mode()
    def _get_score_impl(self, text: List[str]) -> torch.Tensor:
        encodings = tokenize_sequences(text, self.tokenizer, self.observer_model.device)
        observer_logits, performer_logits = self._get_logits(encodings)
        log_ppl = batch_label_cross_entropy(performer_logits, encodings.input_ids)
        x_ppl = batch_cross_entropy(observer_logits, performer_logits.to(self.observer_model.device))
        return log_ppl / x_ppl

    @property
    def threshold(self) -> float:
        if self.scoring_mode == 'low-fpr':
            return self.BINOCULARS_FPR_THRESHOLD
        if self.scoring_mode == 'accuracy':
            return self.BINOCULARS_ACCURACY_THRESHOLD
        raise ValueError(f'Invalid scoring mode: {self.scoring_mode}')

    @torch.inference_mode()
    def _predict_impl(self, text: List[str]) -> torch.Tensor:
        return self._get_score_impl(text) < self.threshold
