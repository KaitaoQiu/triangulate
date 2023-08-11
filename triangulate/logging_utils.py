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

from typing import Any

from rich.console import Console
from rich.panel import Panel


CONSOLE = Console()

STEP_STYLE = "bold blue"
ACTION_STYLE = "bold purple4"


def print_panel(renderable: Any, title: str = ""):
  CONSOLE.print(Panel(renderable, title=title))


def print_horizontal_line(title: str = ""):
  CONSOLE.rule(title=title)
