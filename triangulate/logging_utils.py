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

"""Printing and logging utilities."""

import shutil
import termcolor


def print_color(
    prompt: str,
    message: str = "",
    *,
    color: int | str | None = None,
    bold: bool = True,
):
  """Prints a prompt with optional message and formatting options."""
  attrs = []
  if bold:
    attrs.append("bold")
  if message:
    prompt = f"{prompt}: "
  prompt_colored = termcolor.colored(prompt, color=color, attrs=attrs)
  if message:
    print(f"{prompt_colored}{message}")
  else:
    print(prompt_colored)


def print_horizontal_line():
  terminal_size = shutil.get_terminal_size()
  print("\u2500" * terminal_size.columns)
