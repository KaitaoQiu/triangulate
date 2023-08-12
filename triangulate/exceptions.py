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

"""Bug localization exceptions."""

from collections.abc import Sequence
import dataclasses
import os

import ordered_set
from python_graphs import program_graph

from triangulate import str_utils

OrderedSet = ordered_set.OrderedSet
ProgramGraph = program_graph.ProgramGraph
ProgramGraphNode = program_graph.ProgramGraphNode


class BugLocalizationError(Exception):
  """A bug localization error."""


@dataclasses.dataclass
class SubjectProgramNoExceptionRaisedError(BugLocalizationError):
  """Subject program does not raise an exception."""

  subject: os.PathLike[str]
  subject_argv: Sequence[str]


@dataclasses.dataclass
class LineNumberOutOfBoundsError(BugLocalizationError):
  """Line number is out of bounds for program."""

  code: str
  lineno: int

  def __str__(self) -> str:
    return (
        f"Line number {self.lineno} is out of bounds for code:\n"
        f"{str_utils.prepend_line_numbers(self.code)}"
    )


@dataclasses.dataclass
class CouldNotIdentifyIllegalStateExpressionError(BugLocalizationError):
  """Could not identify illegal state expression from code and line number."""

  code: str
  lineno: int


@dataclasses.dataclass
class VariableAlreadyInspectedError(BugLocalizationError):
  """Cannot inspect variable that has already been inspected."""

  variable: ProgramGraphNode
  inspected_variables: OrderedSet[ProgramGraphNode]
  candidate_variables_to_inspect: OrderedSet[ProgramGraphNode]


@dataclasses.dataclass
class VariableNotInspectionCandidateError(BugLocalizationError):
  """Cannot inspect variable that is not a candidate for inspection."""

  variable: ProgramGraphNode
  inspected_variables: OrderedSet[ProgramGraphNode]
  candidate_variables_to_inspect: OrderedSet[ProgramGraphNode]
