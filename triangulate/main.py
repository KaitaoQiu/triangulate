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

import runpy
import sys
import traceback

from absl import app
from absl import flags
from triangulate import core
from triangulate import logging_utils

Localiser = core.Localiser
Environment = core.Environment

print_color = logging_utils.print_color
print_horizontal_line = logging_utils.print_horizontal_line

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

  try:
    print_color(prompt="Running", message=subject, color="blue")
    print_horizontal_line()

    # Run Python program.
    runpy.run_path(subject, run_name="__main__")

    print_horizontal_line()
    print_color(prompt="Success", message=subject, color="green")
    print_color(
        prompt="Triangulate did not run because no exception was thrown.",
        color="green",
    )
  except Exception as e:  # pylint: disable=broad-except
    print_horizontal_line()
    print_color(prompt="Exception caught:", color="yellow")
    _, _, tb = sys.exc_info()
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
      print_color(
          "Could not resolve illegal state expression from exception:",
          color="red",
      )
      traceback.print_exception(e, limit=-1)


if __name__ == "__main__":
  app.run(main)
