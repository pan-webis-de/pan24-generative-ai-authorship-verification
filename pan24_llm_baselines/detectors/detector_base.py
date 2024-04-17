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

from abc import ABC, abstractmethod
from typing import Iterable, List, Union


class DetectorBase(ABC):
    """
    LLM detector base class.
    """

    @abstractmethod
    def _get_score_impl(self, text: List[str]) -> Iterable[float]:
        pass

    def get_score(self, text: Union[str, List[str]]) -> Union[float, Iterable[float]]:
        """
        Return a prediction score indicating the "humanness" of the input text.

        :param text: input text or list of input texts
        :return: humanness score(s)
        """
        return_str = isinstance(text, str)
        if return_str:
            text = [text]
        text = self._get_score_impl(text)
        return text[0] if return_str else text

    @abstractmethod
    def predict(self, text: Union[str, List[str]]) -> Union[bool, Iterable[bool]]:
        """
        Make a prediction whether the input text was written by a human.

        :param text: input text or list of input texts
        :return: boolean values indicating whether inputs are classified as human
        """
