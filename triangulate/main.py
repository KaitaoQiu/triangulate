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

from absl import app
from absl import flags
from absl import logging
from triangulate import core

Localiser = core.Localiser
Environment = core.Environment

_SUBJECT = flags.DEFINE_string(
    "subject",
    None,
    help="the name of a buggy file",
    required=True,
    short_name="p",
)
_BUG_TRIGGERING_INPUT = flags.DEFINE_string(
    "bug_triggering_input",
    None,
    required=True,
    short_name="b",
    help="a bug-triggering input",
)
_LOGLEVEL = flags.DEFINE_integer(
    "loglevel",
    0,
    short_name="l",
    help="Set logging level (default: INFO)",
)
_BUG_TRAP = flags.DEFINE_integer(
    "bug_trap",
    0,
    short_name="t",
    help="Program line at which the bug was observed",
)
# During burnin, the program stores outputs for later use to checking
# whether injecting/executing probes has changed program semantics.
_BURNIN = flags.DEFINE_integer(
    "burnin",
    0,
    short_name="n",
    help=(
        "Percentage of max_steps to use as burnin steps "
        "to tolerate nondeterministic buggy programs; "
        "zero (the default) disables burnin."
    ),
)
_MAX_STEPS = flags.DEFINE_integer(
    "max_steps",
    10,
    short_name="m",
    help="maximum simulation steps",
)
_PROBE_OUTPUT_FILENAME = flags.DEFINE_string(
    "probe_output_filename",
    "__probeOutput.dmp",
    short_name="o",
    help="maximum simulation steps",
)


def main(argv):
  """Program entry point."""

  if len(argv) < 1:
    raise app.UsageError("Too few command-line arguments.")

  logging.set_verbosity(_LOGLEVEL.value)

  if not 0 <= _BURNIN.value < 1:
    err_template = "Error: burnin period must fall into the interval [0,1)."
    logging.error(err_template)
    raise ValueError(err_template)

  if _SUBJECT.present:
    subject = _SUBJECT.value
  else:
    subject = input("Please enter the name of the buggy program: ")

  env = Environment(
      subject=subject,
      bug_triggering_input=_BUG_TRIGGERING_INPUT.value,
      bug_trap=_BUG_TRAP.value,
      burnin=_BURNIN.value,
      max_steps=_MAX_STEPS.value,
      probe_output_filename=_PROBE_OUTPUT_FILENAME.value,
      loglevel=_LOGLEVEL.value,
  )
  localiser = Localiser(env)

  while not env.terminate():
    env.update(localiser.pick_action(env.state, env.reward()))


if __name__ == "__main__":
  app.run(main)
