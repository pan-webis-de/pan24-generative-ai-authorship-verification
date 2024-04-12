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

from typing import Iterable

from pan24_llm_baselines.detectors.detector_base import DetectorBase
from pan24_llm_baselines.perturbators.perturbator_base import PerturbatorBase
from pan24_llm_baselines.perturbators.t5_mask import T5MaskPerturbator
from pan24_llm_baselines.torch_util import *


class DetectGPT(DetectorBase):
    """
    DetectGPT LLM detector.

    This is a reimplementation of the original: https://github.com/eric-mitchell/detect-gpt

    References:
    ===========
        Mitchell, Eric, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning,
        and Chelsea Finn. 2023. “DetectGPT: Zero-Shot Machine-Generated Text
        Detection Using Probability Curvature.” arXiv [Cs.CL]. arXiv.
        http://arxiv.org/abs/2301.11305.
    """
    def __init__(self, base_model='tiiuae/falcon-7b',
                 device: TorchDeviceMapType = 'auto',
                 perturbator: PerturbatorBase = None,
                 n_perturbations=50,
                 **base_model_args):
        """
        :param base_model: base language model
        :param device: base model device
        :param perturbator: perturbation model (default: T5MaskPerturbator)
        :param base_model_args: additional base model arguments
        """

        self.n_perturbations = n_perturbations
        self.perturbator = perturbator if perturbator else T5MaskPerturbator(device=device)
        self.base_model = transformers_load_model(
            base_model,
            device_map=device,
            **base_model_args)
        self.base_tokenizer = transformers_load_tokenizer(base_model)

    @torch.inference_mode()
    def get_score(self, text: Union[str, List[str]]) -> Union[float, Iterable[float]]:
        enc_orig = torch_tokenize(text, self.base_tokenizer, self.base_model.device, 512)
        ppl_orig = -torch_perplexity(self.base_model, enc_orig).cpu().item()

        perturbed = self.perturbator.perturb(text, n_variants=self.n_perturbations)
        enc_pert = torch_tokenize(perturbed, self.base_tokenizer, self.base_model.device, 512)
        ppl_pert = -torch_perplexity(self.base_model, enc_pert).cpu()
        ppl_pert_std = ppl_pert.std().item()
        ppl_pert = ppl_pert.mean().item()

        return max(0.0, (ppl_orig - ppl_pert) / ppl_pert_std)

    def predict(self, text: Union[str, List[str]]) -> Union[bool, Iterable[bool]]:
        pass
