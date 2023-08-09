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

"""Main script."""

import contextlib
import io
import runpy
import sys
import traceback
from types import TracebackType
from typing import TypeAlias

from absl import app
from absl import flags
from triangulate import core
from triangulate import logging_utils

ExcInfo: TypeAlias = tuple[type[BaseException], BaseException, TracebackType]

Localiser = core.Localiser
Environment = core.Environment

CONSOLE = logging_utils.CONSOLE
print_panel = logging_utils.print_panel

# During burnin, the program stores outputs for later use to checking
# whether injecting/executing probes has changed program semantics.
_BURNIN_STEPS = flags.DEFINE_integer(
    "burnin_steps",
    None,
    short_name="n",
    help=(
        "Percentage of max_steps to use as burnin steps to tolerate "
        "nondeterministic buggy programs; zero (the default) disables burnin."
    ),
)
_MAX_STEPS = flags.DEFINE_integer(
    "max_steps",
    None,
    short_name="m",
    help="maximum simulation steps",
)


def run_from_exception(
    exc_info: ExcInfo,
    subject: str,
    burnin_steps: int | None,
    max_steps: int | None,
    # exception: Exception,
):
  CONSOLE.print("Exception caught:", style="bold yellow")
  _, exc_value, tb = exc_info
  if exc_value is None:
    raise ValueError("No exception raised")
  traceback.print_tb(tb)
  tb_info = traceback.extract_tb(tb)
  exc_last_frame = tb_info[-1]
  exc_lineno = exc_last_frame.lineno

  try:
    subject_argv = sys.argv
    core.run(
        subject=subject,
        subject_argv=subject_argv,
        bug_lineno=exc_lineno,
        burnin_steps=burnin_steps,
        max_steps=max_steps,
    )
  except core.CouldNotResolveIllegalStateExpressionError:
    CONSOLE.print(
        "Could not resolve illegal state expression from exception:",
        style="bold red",
    )
    traceback.print_exception(exc_value, limit=-1)


def main(argv):
  if len(argv) < 2:
    raise app.UsageError(
        "Usage: triangulate [flags...] subject -- [subject_args...]"
    )

  subject = argv[1]

  # Rewrite `sys.argv` to absl-parsed `argv`.
  # New `sys.argv`: <subject> <subject arguments>...
  this_module_name = sys.argv[0]
  sys.argv = argv[1:]

  # Save flag values.
  burnin_steps = _BURNIN_STEPS.value
  max_steps = _MAX_STEPS.value

  # Remove parsed flags to avoid flag name conflicts with the subject module.
  flag_module_dict = flags.FLAGS.flags_by_module_dict()
  fv = flags.FlagValues()
  for flag in flag_module_dict[this_module_name]:
    fv[flag.name] = flag
  flags.FLAGS.remove_flag_values(fv)

  buffer = io.StringIO()
  exc_info = None
  try:
    CONSOLE.print(rf"[blue][b]Running:[/b][/blue] {subject}")
    with (
        contextlib.redirect_stdout(buffer),
        contextlib.redirect_stderr(buffer),
    ):
      runpy.run_path(subject, run_name="__main__")
  except Exception:  # pylint: disable=broad-except
    exc_info = sys.exc_info()
  finally:
    print_panel(buffer.getvalue().removesuffix("\n"), title="Subject output")

  if exc_info is not None:
    run_from_exception(
        exc_info=exc_info,
        subject=subject,
        burnin_steps=burnin_steps,
        max_steps=max_steps,
    )
    return

  CONSOLE.print(rf"[green][b]Success:[/b][/green] {subject}")
  CONSOLE.print(
      "Triangulate did not run because no exception was thrown.",
      style="bold green",
  )


if __name__ == "__main__":
  app.run(main)
