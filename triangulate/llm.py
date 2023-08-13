# Copyright 2023 The triangulate Authors.
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

"""LLM support.

Reads configuration variables from .env or the environment (`os.environ`).

Example usages:
- python3 -m fire triangulate.llm generate palm_text <prompt>
- python3 -m fire triangulate.llm list_palm_models
"""

import enum
import functools
import os
from typing import Sequence

import dotenv
import fire
from google.api_core import retry
import google.generativeai as palm

from triangulate import environment
from triangulate import logging_utils

State = environment.State
rprint = logging_utils.rprint

# Configuration: inject .env configuration variables into `os.environ`.
dotenv.load_dotenv(override=True)
TIMEOUT_SECONDS = os.environ.get('TRIANGULATE_LLM_TIMEOUT_SECONDS', 20)


class ModelType(enum.Enum):
  PALM_TEXT = 'models/text-bison-001'
  PALM_CHAT = 'models/chat-bison-001'


@functools.singledispatch
def generate(model: ModelType, prompt: str) -> str:
  """Generates text via a LLM."""
  match model:
    case ModelType.PALM_TEXT:
      return generate_palm_text(prompt=prompt, model=model)
    case _:
      raise ValueError(f'Unknown model type: {model}')


@generate.register
def _(model: str, prompt: str) -> str:
  model_type = ModelType[model.upper()]
  return generate(model_type, prompt)


################################################################################
# PaLM
################################################################################


@functools.cache
def configure_palm():
  """Configures the PaLM API key, if not already configured. Only runs once."""
  palm_api_key = os.environ['PALM_API_KEY']
  palm.configure(api_key=palm_api_key)


def list_palm_models():
  """Print available PaLM models."""
  configure_palm()
  for model in palm.list_models():
    rprint(model)


@retry.Retry(deadline=TIMEOUT_SECONDS)
def generate_palm_text(
    prompt: str,
    model: ModelType | str | None = None,
    temperature: float | None = None,
    candidate_count: int | None = None,
    max_output_tokens: int | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    stop_sequences: str | Sequence[str] | None = None,
) -> str:
  """Generates text via a PaLM text model."""
  configure_palm()
  if isinstance(model, ModelType):
    model_name = model.value
  else:
    model_name = model
  completion = palm.generate_text(
      model=model_name,
      prompt=prompt,
      temperature=temperature,
      candidate_count=candidate_count,
      max_output_tokens=max_output_tokens,
      top_p=top_p,
      top_k=top_k,
      stop_sequences=stop_sequences,
  )
  return completion.result


if __name__ == '__main__':
  fire.Fire()
