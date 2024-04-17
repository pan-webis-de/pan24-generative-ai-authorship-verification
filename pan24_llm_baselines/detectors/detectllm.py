from typing import List, Literal

from pan24_llm_baselines.detectors.detectgpt import DetectGPT
from pan24_llm_baselines.perturbators.perturbator_base import PerturbatorBase
from pan24_llm_baselines.util import *


class DetectLLM(DetectGPT):
    """
    DetectLLM LLM detector.

    This is a reimplementation of the original: https://github.com/mbzuai-nlp/DetectLLM

    References:
    ===========
        Su, Jinyan, Terry Yue Zhuo, Di Wang, and Preslav Nakov. 2023. “DetectLLM: Leveraging
        Log Rank Information for Zero-Shot Detection of Machine-Generated Text.” arXiv [Cs.CL].
        arXiv. http://arxiv.org/abs/2306.05540.
    """
    def __init__(self,
                 scoring_mode: Literal['lrr', 'npr'] = 'lrr',
                 base_model='tiiuae/falcon-7b',
                 device: TorchDeviceMapType = 'auto',
                 perturbator: PerturbatorBase = None,
                 n_samples=20,
                 batch_size=10,
                 verbose=True,
                 **base_model_args):
        """
        :param scoring_mode: ``'lrr'`` (Log-Likelihood Log-Rank Ratio) or
                             ``'npr'`` (Normalized Log-Rank Perturbation)
        :param base_model: base language model
        :param device: base model device
        :param perturbator: perturbation model for NPR (default: T5MaskPerturbator)
        :param n_samples: number of perturbed texts to generate for NPR
        :param batch_size: Log-likelihood prediction batch size
        :param verbose: show progress bar
        :param base_model_args: additional base model arguments
        """
        super().__init__(base_model, device, perturbator, n_samples,
                         batch_size, verbose, **base_model_args)
        self.scoring_mode = scoring_mode

    def _get_logits(self, text: List[str], verbose_msg: str):
        encoding = tokenize_sequences(text, self.base_tokenizer, self.base_model.device, 512)
        logits = [l.cpu() for l, _ in model_batch_forward(self.base_model, encoding, self.batch_size, verbose_msg)]
        return torch.vstack(logits), encoding.input_ids.cpu()

    def _lrr(self, text: List[str]) -> List[float]:
        verbose_msg = 'Calculating logits' if self.verbose else None
        logits, labels = self._get_logits(text, verbose_msg)
        ll = batch_label_cross_entropy(logits, labels)
        lrr = batch_label_log_rank(logits, labels)
        return (ll / lrr).tolist()

    def _npr(self, text: List[str]) -> Iterable[float]:
        verbose_msg = 'Calculating original logits' if self.verbose else None
        logits, labels = self._get_logits(text, verbose_msg)
        orig_rank = batch_label_log_rank(logits, labels)

        perturbed = self.perturbator.perturb(text, n_variants=self.n_samples)
        verbose_msg = 'Calculating perturbed logits' if self.verbose else None
        logits, labels = self._get_logits(perturbed, verbose_msg)
        pert_rank = batch_label_log_rank(logits, labels).mean(-1)

        return (pert_rank / orig_rank).tolist()

    @torch.inference_mode()
    def _get_score_impl(self, text: List[str]) -> Iterable[float]:
        if self.scoring_mode == 'lrr':
            return self._lrr(text)
        return self._npr(text)
