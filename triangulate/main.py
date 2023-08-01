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

flags.DEFINE_string(
    "subject",
    None,
    help="the name of a buggy file",
    required=True,
    short_name="p",
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

  FLAGS = flags.FLAGS # pylint: disable=invalid-name
  logging.set_verbosity(FLAGS.loglevel)

  if not 0 <= FLAGS.burnin < 1:
    err_template = "Error: burnin period must fall into the interval [0,1)."
    logging.error(err_template)
    raise ValueError(err_template)

  if not FLAGS.subject:
    FLAGS.subject = input(
        "Please enter the name of the subject program: "
    )

  # Cannot use ** as Abseil populates its dictionary with unrelated flags
  flags_dict = FLAGS.flag_values_dict()
  env = Environment(
    subject = flags_dict["subject"],
    bug_triggering_input = flags_dict["bug_triggering_input"],
    bug_trap = flags_dict["bug_trap"],
    burnin = flags_dict["burnin"],
    max_steps = flags_dict["max_steps"],
    probe_output_filename = flags_dict["probe_output_filename"],
    loglevel = FLAGS.loglevel
  )
  localiser = Localiser(env)

  while not env.terminate():
    env.update(localiser.pick_action(env.state, env.reward()))

if __name__ == "__main__":
  app.run(main)
