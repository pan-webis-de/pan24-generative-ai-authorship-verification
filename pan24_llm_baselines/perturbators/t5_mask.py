import random
from typing import Tuple

import numpy as np
import numpy.typing as npt
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM

from pan24_llm_baselines.torch_util import *
from pan24_llm_baselines.perturbators.perturbator_base import PerturbatorBase


class T5MaskPerturbator(PerturbatorBase):
    def __init__(self,
                 mask_span_length=2,
                 mask_pct=0.15,
                 mask_buffer_size=1,
                 model_name='t5-large',
                 device: TorchDeviceMapType = 'auto',
                 quantization_bits=None,
                 use_flash_attn=False,
                 max_tokens=512,
                 batch_size=20,
                 verbose=True,
                 **model_args):
        """
        :param mask_span_length: length of token spans to mask out
        :param mask_pct: target percentage of tokens to mask out
        :param buffer_size: minimum buffer around mask tokens
        :param model_name: T5 model variant
        :param device: model device
        :param quantization_bits: quantize model
        :param use_flash_attn: use flash attention
        :param max_tokens: max number of tokens the model can handle
        :param batch_size: perturbation variant batch size
        :param verbose: show progress bar
        :param model_args: additional model arguments
        """
        self.span_length = mask_span_length
        self.mask_pct = mask_pct
        self.mask_buffer_size = mask_buffer_size
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.verbose = verbose

        self.model = transformers_load_model(model_name,
                                             device_map=device,
                                             auto_cls=AutoModelForSeq2SeqLM,
                                             quantization_bits=quantization_bits,
                                             use_flash_attn=use_flash_attn,
                                             **model_args)
        self.tokenizer = transformers_load_tokenizer(model_name, model_max_length=max_tokens)

    def _mask_tokens(self, token_ids, n_target) -> Tuple[npt.NDArray[np.float64], int]:
        """
        Randomly replace token spans with masks.

        :param token_ids: input token ids
        :param n_target: target number of masks
        :return: masked token sequence, number of masked tokens
        """
        mask_id = self.tokenizer.encode('<extra_id_0>')[0]

        spans = []
        while len(spans) < n_target:
            start = random.randint(0, len(token_ids) - self.span_length)
            end = start + self.span_length
            if mask_id not in token_ids[max(0, start - self.mask_buffer_size):end + self.mask_buffer_size]:
                token_ids[start:end] = mask_id
                spans.append(start)

        spans = sorted(spans)
        for i, idx in enumerate(spans):
            # count down from <extra_id_0> to <extra_id_{len(spans)}>
            token_ids[idx] -= i

        # Delete remainder of the spans
        del_indexes = np.hstack([np.arange(s + 1, min(s + self.span_length + 1, len(token_ids))) for s in spans])
        return np.delete(token_ids, del_indexes), len(spans)

    def _generate_text(self, token_ids: torch.Tensor, num_masks: int):
        """
        Generate a new text from masked token sequence.
        
        :param token_ids: (batch of) masked input token ids
        :param num_masks: number of masks in the text
        :return: generated text
        """
        if len(token_ids.shape) < 2:
            token_ids = token_ids.reshape((1, *token_ids.shape))

        stop_id = self.tokenizer.encode(f'<extra_id_{num_masks}>')[0]
        outputs = self.model.generate(
            input_ids=token_ids.to(self.model.device),
            attention_mask=torch.ones(token_ids.shape, device=self.model.device),
            max_length=512,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=stop_id,
            pad_token_id=self.tokenizer.pad_token_id).cpu()

        out_ids_batch = []
        for output in outputs:
            if len(token_ids) > 1:
                output = output[output != self.tokenizer.pad_token_id]

            token_ids = token_ids.flatten()
            input_mask_pos = np.argwhere(token_ids >= stop_id).flatten()
            output_mask_pos = torch.argwhere(output >= stop_id).flatten().cpu()
            last_pos = 0
            out_ids = []
            for i in range(min(len(input_mask_pos), len(output_mask_pos))):
                out_ids.extend(token_ids[last_pos:input_mask_pos[i]])
                out_ids.extend(output[output_mask_pos[i] + 1:output_mask_pos[min(i + 1, len(output_mask_pos) - 1)]])
                last_pos = input_mask_pos[i] + 1
            out_ids.extend(token_ids[last_pos:-1])
            out_ids_batch.append(out_ids)

        return self.tokenizer.batch_decode(out_ids_batch)

    def perturb(self, text: str, n_variants: int = 1) -> Union[str, List[str]]:
        tokens_batch = torch_tokenize(text, self.tokenizer, max_length=self.max_tokens, return_tensors='np').input_ids
        perturbed_texts = []

        for token_ids in tokens_batch:
            masked = []
            n_masks = min(int(self.mask_pct * len(token_ids) / (self.span_length + self.mask_buffer_size * 2)) + 1, 99)
            for _ in range(n_variants):
                t, _ = self._mask_tokens(np.array(token_ids), n_masks)
                masked.append(torch.from_numpy(t))

            batch_it = range(0, len(masked), self.batch_size)
            if self.verbose:
                batch_it = tqdm(batch_it, desc='Generating perturbations', leave=False, unit=' batches')
            for b in batch_it:
                m = pad_sequence(
                    masked[b:b + self.batch_size],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id)
                perturbed_texts.extend(self._generate_text(m, n_masks))

        return perturbed_texts if len(perturbed_texts) > 1 else perturbed_texts[0]
