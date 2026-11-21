#
# Copyright 2026 Volvo Car Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fact-based graph retrieval with configurable LLM filtering and reranking."""

from .beam import Retriever
from .models import Answer, FactEvidence, Passage, RetrievalResult

BeamRetriever = Retriever

__all__ = ["Answer", "BeamRetriever", "FactEvidence", "Passage", "RetrievalResult", "Retriever"]
