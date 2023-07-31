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

import os

from absl import app
from absl import flags
from absl import logging
from triangulate import core

Localiser = core.Localiser
Environment = core.Environment

flags.DEFINE_string(
    "buggy_program_name",
    None,
    help="the name of a buggy file",
    required=True,
    short_name="p",
)
flags.DEFINE_string(
    "illegal_state_expr",
    None,
    required=True,
    short_name="i",
    help=(
        "An expression defining illegal state; it is a fragment "
        "of the program's specification, which is almost never "
        "fully realised. Concretely, it will, for us, usually "
        "be the complement of an assertion."
    ),
)
flags.DEFINE_string(
    "bug_triggering_input",
    None,
    required=True,
    short_name="b",
    help="a bug-triggering input",
)
flags.DEFINE_integer(
    "loglevel",
    0,
    short_name="l",
    help="Set logging level (default: INFO)",
)
flags.DEFINE_integer(
    "bug_trap",
    0,
    short_name="t",
    help="Program line at which the bug was observed",
)
# During burnin, the program stores outputs for later use to checking
# whether injecting/executing probes has changed program semantics.
flags.DEFINE_integer(
    "burnin",
    0,
    short_name="n",
    help=(
        "Percentage of max_steps to use as burnin steps "
        "to tolerate nondeterministic buggy programs; "
        "zero (the default) disables burnin."
    ),
)
flags.DEFINE_integer(
    "max_steps",
    10,
    short_name="m",
    help="maximum simulation steps",
)
flags.DEFINE_string(
    "probe_output_filename",
    "__probeOutput.dmp",
    short_name="o",
    help="maximum simulation steps",
)


def main(argv):
  """Program entry point."""

  if len(argv) < 1:
    raise app.UsageError("Too few command-line arguments.")

  FLAGS = flags.FLAGS
  logging.set_verbosity(FLAGS.loglevel)

  if not 0 <= FLAGS.burnin < 1:
    err_template = "Error: burnin period must fall into the interval [0,1)."
    logging.error(err_template)
    raise ValueError(err_template)

  if not FLAGS.buggy_program_name:
    FLAGS.buggy_program_name = input(
        "Please enter the name of the buggy program: "
    )

  env = Environment(**FLAGS.flag_values_dict())
  localiser = Localiser(env)

  while not env.terminate():
    env.update(localiser.pick_action(env.state, env.reward()))

  if FLAGS.loglevel == logging.DEBUG:
    # The following statement exploits an undocumented feature of Python 3.x.
    env.subject_with_probes._closer.delete = False 
    print(
        f"The instrumented subject program saved to {env.subject_with_probes.name}."
    )

if __name__ == "__main__":
  app.run(main)
