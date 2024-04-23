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

from typing import List

import torch

from pan24_llm_baselines.detectors.detector_base import DetectorBase
from pan24_llm_baselines.util import *

__all__ = ['DegenerateLLM']


class DegenerateLLM(DetectorBase):
    """
    Text degeneration LLM detector.
    """

    def __init__(self,
                 base_model='openai-community/gpt2',
                 device: TorchDeviceMapType = 'auto',
                 prefix_length=128,
                 max_generation_length=256,
                 use_flash_attn=False,
                 quantization_bits=None,
                 **model_args):
        """
        :param base_model: base generation model
        :param device: model device
        :param prefix_length: max number of tokens to use as input prefix (minimum: 64)
        :param max_generation_length: max number of tokens to generate (minimum: 64)
        :param use_flash_attn: use flash attention
        :param quantization_bits: quantize model
        :param model_args: additional model args
        """

        self.base_model = load_model(
            base_model,
            device_map=device,
            use_flash_attn=use_flash_attn,
            quantization_bits=quantization_bits,
            **model_args)
        self.prefix_len = max(64, prefix_length)
        self.max_generation_len = max(64, max_generation_length)
        self.tokenizer = load_tokenizer(base_model,
                                        max_length=self.prefix_len + self.max_generation_len,
                                        padding_side='left')

    def _normalize_scores(self, scores):
        return torch.sigmoid(2 * (scores.to(torch.float64) - 6))

    @torch.inference_mode()
    def _get_score_impl(self, text: List[str]) -> torch.Tensor:
        encoding = tokenize_sequences(text, self.tokenizer, self.base_model.device)
        if encoding.input_ids.shape[1] < self.prefix_len + 50:
            raise ValueError(f'Inputs must be at least {self.prefix_len + 50} tokens long.')

        suffix_len = min(self.max_generation_len, encoding.input_ids.shape[1] - self.prefix_len)
        generation = self.base_model.generate(input_ids=encoding.input_ids[:self.prefix_len],
                                              attention_mask=encoding.attention_mask[:self.prefix_len],
                                              do_sample=False,
                                              max_new_tokens=suffix_len,
                                              pad_token_id=self.tokenizer.eos_token_id,
                                              num_return_sequences=1,
                                              return_dict_in_generate=True,
                                              output_logits=True)
        gen_logits = torch.stack(generation.logits, dim=1)

        suffix_len = gen_logits.shape[1]
        total_len = self.prefix_len + suffix_len

        suffix_ids = encoding.input_ids[:, self.prefix_len:total_len]
        suffix_mask = encoding.attention_mask[:, self.prefix_len:total_len]
        ce_pred = seq_label_cross_entropy(gen_logits, suffix_ids, suffix_mask, shift=False).cpu()

        truth_ids = encoding.input_ids[:total_len]
        truth_mask = encoding.attention_mask[:total_len]
        truth_logits = self.base_model(input_ids=truth_ids, attention_mask=truth_mask).logits
        ce_truth = seq_label_cross_entropy(truth_logits, truth_ids, truth_mask).cpu()

        return ce_pred / ce_truth
