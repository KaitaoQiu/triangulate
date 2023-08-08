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

"""Probing instrumentation utilities."""

from collections.abc import Mapping, Sequence
import contextlib
import inspect
import io
import runpy
import sys
import tempfile
import traceback
from typing import Any

from triangulate import logging_utils

print_color = logging_utils.print_color

# A unique name for the probing function.
PROBE_FUNCTION_NAME = "triangulate_probe"


def make_probe_call(variable_names: Sequence[str]) -> str:
  return f"{PROBE_FUNCTION_NAME}({variable_names}, locals())\n"


def probe(variable_names: Sequence[str], locals_dict: Mapping[str, Any]):
  """Probes variables and prints their values."""
  caller = inspect.getframeinfo(inspect.stack()[1][0])
  full_probe_str = io.StringIO()
  for name in variable_names:
    value = locals_dict.get(name, None)
    if value is None:
      continue
    probe_str = f"{name} = {repr(value)}"
    full_probe_str.write(probe_str)
    full_probe_str.write("\n")
  if not full_probe_str.getvalue():
    return
  print_color(
      prompt=f"Probing: {caller.filename}:{caller.lineno}",
      color="blue",
  )
  print(full_probe_str.getvalue(), end="")


def run_with_instrumentation(
    python_source: str,
    argv: Sequence[str],
) -> str:
  """Executes `python_source` and returns stdout and stderr concatenated."""
  # A `globals()` dict for usage with `exec` to support instrumentation.
  exec_globals = {PROBE_FUNCTION_NAME: probe}
  buffer = io.StringIO()
  with (
      contextlib.redirect_stdout(buffer),
      contextlib.redirect_stderr(buffer),
  ):
    old_argv = sys.argv
    try:
      with tempfile.NamedTemporaryFile(mode="w+t") as f:
        f.write(python_source)
        f.flush()
        sys.argv = list(argv)
        runpy.run_path(f.name, init_globals=exec_globals, run_name="__main__")
    except Exception as e:  # pylint:disable=broad-except
      # Print exception from the executed program.
      print_color("Exception raised:", color="yellow", bold=False)
      traceback.print_exception(e.__context__ or e)
    finally:
      sys.argv = old_argv
  return buffer.getvalue()
