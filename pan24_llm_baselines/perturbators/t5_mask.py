import random
from typing import Tuple

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM

from pan24_llm_baselines.util import *
from pan24_llm_baselines.perturbators.perturbator_base import PerturbatorBase


class T5MaskPerturbator(PerturbatorBase):
    def __init__(self,
                 span_length=2,
                 mask_pct=0.3,
                 buffer_size=1,
                 model_name='t5-large',
                 device: TorchDeviceMapType = 'auto',
                 quantization_bits=None,
                 use_flash_attn=False,
                 max_tokens=512,
                 batch_size=5,
                 verbose=True,
                 **model_args):
        """
        :param span_length: length of token spans to mask out
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
        self.span_length = span_length
        self.mask_pct = mask_pct
        self.mask_buffer_size = buffer_size
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.verbose = verbose

        self.model = load_model(model_name,
                                device_map=device,
                                auto_cls=AutoModelForSeq2SeqLM,
                                quantization_bits=quantization_bits,
                                use_flash_attn=use_flash_attn,
                                **model_args)
        self.tokenizer = load_tokenizer(model_name, model_max_length=max_tokens)

    def _mask_tokens(self, token_ids) -> Tuple[npt.NDArray[np.float64], int]:
        """
        Randomly replace token spans with masks.
        Strips padding tokens from the sequence.

        :param token_ids: input token ids
        :return: masked token sequence, number of masked tokens
        """
        mask_id = self.tokenizer.encode('<extra_id_0>')[0]
        n_padding = np.sum(token_ids == self.tokenizer.pad_token_id)
        text_len = len(token_ids) - n_padding
        n_spans_target = int(self.mask_pct * text_len / (self.span_length + self.mask_buffer_size * 2) + 1)
        n_spans_target = min(n_spans_target, 99)    # T5 has a max of 100 sentinel tokens by default

        spans = []
        while len(spans) < n_spans_target:
            start = random.randint(0, text_len - self.span_length - 1)
            end = start + self.span_length
            if mask_id not in token_ids[max(0, start - self.mask_buffer_size):end + self.mask_buffer_size]:
                token_ids[start:end] = mask_id
                spans.append(start)

        spans = sorted(spans)
        for i, idx in enumerate(spans):
            # count down from <extra_id_0> to <extra_id_{len(spans)}>
            token_ids[idx] -= i

        # Delete remainder of the spans
        del_indexes = np.hstack([np.arange(s + 1, s + self.span_length) for s in spans])
        return np.delete(token_ids, del_indexes), len(spans)

    def _generate_texts(self, token_ids: torch.Tensor, num_masks: Union[List[int], torch.Tensor]):
        """
        Generate a new texts from batch of masked token sequence.
        
        :param token_ids: (batch of) masked input token ids
        :param num_masks: number of masks for each text
        :return: generated text
        """
        if len(token_ids.shape) < 2:
            token_ids = token_ids.reshape((1, *token_ids.shape))

        stop_ids = torch.tensor([self.tokenizer.encode(f'<extra_id_{n}>')[0] for n in num_masks])
        outputs = self.model.generate(
            input_ids=token_ids.to(self.model.device),
            attention_mask=torch.ones(token_ids.shape, device=self.model.device),
            max_length=min(512, max(num_masks) * 8),
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=stop_ids,
            pad_token_id=self.tokenizer.pad_token_id).cpu()

        out_ids = []
        for i, output in enumerate(outputs):
            if len(token_ids) > 1:
                # Truncate padded </s> tokens
                output = output[output != self.tokenizer.eos_token_id]

            input_mask_pos = torch.argwhere(token_ids[i] >= stop_ids[i])
            output_mask_pos = torch.argwhere(output >= stop_ids[i])
            last_pos = 0
            o = []
            for j in range(min(len(input_mask_pos), len(output_mask_pos))):
                o.extend(token_ids[i][last_pos:input_mask_pos[j]])
                o.extend(output[output_mask_pos[j] + 1:output_mask_pos[min(j + 1, len(output_mask_pos) - 1)]])
                last_pos = input_mask_pos[j] + 1
            o.extend(token_ids[i][last_pos:-1])
            out_ids.append(o)
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

    def perturb(self, text: Union[str, List[str]], n_variants: int = 1) -> Union[str, List[str]]:
        batch_size = self.batch_size if self.batch_size else len(text)
        if isinstance(text, str):
            text = [text]

        perturbed = []
        batch_it = range(0, len(text), self.batch_size)
        if self.verbose:
            batch_it = tqdm(batch_it, desc='Generating perturbations', leave=False, unit=' batches')
        for b in batch_it:
            tokens_batch = tokenize_sequences(text[b:b + batch_size], self.tokenizer,
                                              max_length=self.max_tokens, return_tensors='np').input_ids
            masked = []
            n_masks = []
            for token_ids in tokens_batch:
                for _ in range(n_variants):
                    t, n = self._mask_tokens(np.array(token_ids))
                    masked.append(torch.from_numpy(t))
                    n_masks.append(n)
            perturbed.extend(self._generate_texts(torch.stack(masked), n_masks))
        return perturbed[0] if n_variants == 1 else perturbed
