import random
from typing import Tuple

import numpy as np
import numpy.typing as npt
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
        :param model_args: additional model arguments
        """
        self.span_length = mask_span_length
        self.mask_pct = mask_pct
        self.mask_buffer_size = mask_buffer_size
        self.max_tokens = max_tokens

        self.model = transformers_load_model(model_name,
                                             device_map=device,
                                             auto_cls=AutoModelForSeq2SeqLM,
                                             quantization_bits=quantization_bits,
                                             use_flash_attn=use_flash_attn,
                                             **model_args)
        self.tokenizer = transformers_load_tokenizer(model_name, model_max_length=max_tokens)

    def _mask_tokens(self, token_ids) -> Tuple[npt.NDArray[np.float64], int]:
        """
        Randomly replace token spans with masks.

        :param token_ids: input token ids
        :return: masked token sequence, number of masked tokens
        """
        mask_id = self.tokenizer.encode('<extra_id_0>')[0]

        n_target = min(int(self.mask_pct * len(token_ids) / (self.span_length + self.mask_buffer_size * 2)) + 1, 99)
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
        
        :param token_ids: masked input token ids
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
            pad_token_id=self.tokenizer.pad_token_id)[0].cpu()
        if len(token_ids) > 1:
            outputs = outputs[outputs != self.tokenizer.pad_token_id]

        token_ids = token_ids.flatten()
        input_mask_pos = np.argwhere(token_ids >= stop_id).flatten()
        output_mask_pos = torch.argwhere(outputs >= stop_id).flatten().cpu()
        last_pos = 0
        out_ids = []
        for i in range(min(len(input_mask_pos), len(output_mask_pos))):
            out_ids.extend(token_ids[last_pos:input_mask_pos[i]])
            # out_ids.extend(outputs[output_mask_pos[i] + 1:output_mask_pos[min(i + 1, len(output_mask_pos) - 1)]])
            last_pos = input_mask_pos[i] + 1
        out_ids.extend(token_ids[last_pos:-1])
        return self.tokenizer.decode(out_ids)

    def perturb(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        tokens_batch = torch_tokenize(text, self.tokenizer, return_tensors='np').input_ids
        perturbed_texts = []
        for batch in tokens_batch:
            masked_ids, n_masks = self._mask_tokens(batch)
            perturbed_texts.append(self._generate_text(torch.from_numpy(masked_ids), n_masks))
        return perturbed_texts if len(perturbed_texts) > 1 else perturbed_texts[0]
