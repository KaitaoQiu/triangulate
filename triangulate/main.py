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

import sys
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


def main(argv):
  if len(argv) < 2:
    raise app.UsageError(
        "Usage: triangulate [flags...] subject -- [subject_args...]"
    )

  # Save flag values.
  burnin_steps = _BURNIN_STEPS.value
  max_steps = _MAX_STEPS.value

  # Get subject program and arguments.
  subject = argv[1]
  subject_argv = argv[1:]

  # Remove parsed flags to avoid flag name conflicts with the subject module.
  # Note: there might be a better way to do this (avoid flags from propagating
  # to subject program).
  flag_module_dict = flags.FLAGS.flags_by_module_dict()
  fv = flags.FlagValues()
  this_module_name = sys.argv[0]
  for flag in flag_module_dict[this_module_name]:
    fv[flag.name] = flag
  flags.FLAGS.remove_flag_values(fv)

  core.run(
      subject,
      subject_argv,
      burnin_steps=burnin_steps,
      max_steps=max_steps,
  )


if __name__ == "__main__":
  app.run(main)
