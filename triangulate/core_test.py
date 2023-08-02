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

"""Tests for core."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from triangulate import core

################################################################################
# Test utilities
################################################################################


def get_first_line_number_by_prefix(s: str, prefix: str) -> int | None:
  """Returns the first line number in `s` whose line starts with `prefix`."""
  for i, line in enumerate(s.splitlines()):
    if line.startswith(prefix):
      return i
  return None


TESTDATA_DIRECTORY = os.path.join(
    absltest.get_default_test_srcdir(),
    'triangulate/testdata',
)
SUBJECT_FILENAME = 'quoter.py'
SUBJECT_FILEPATH = os.path.join(TESTDATA_DIRECTORY, SUBJECT_FILENAME)
with open(SUBJECT_FILEPATH, 'r', encoding='utf8') as f:
  SUBJECT_CONTENT = f.read()

# Note: hardcoded line numbers are unstable between internal and external code.
BUG_TRAP = get_first_line_number_by_prefix(SUBJECT_CONTENT, 'assert')

################################################################################
# Test cases
################################################################################


class EnvironmentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'test_a',
          'subject': SUBJECT_FILEPATH,
          'bug_triggering_input': '42',
          'bug_trap': BUG_TRAP,
          'action': '<placeholder>',
          'expected_output': """\
Today's inspirational quote:
"Believe you can and you're halfway there." - Theodore Roosevelt
""",
      },
      {
          'testcase_name': 'test_b',
          'subject': SUBJECT_FILEPATH,
          'bug_triggering_input': '42',
          'bug_trap': BUG_TRAP,
          'action': '<placeholder>',
          'expected_output': """\
Today's inspirational quote:
"Believe you can and you're halfway there." - Theodore Roosevelt
""",
      },
  )
  def test_execute_and_update(
      self,
      subject: str,
      bug_triggering_input: str,
      bug_trap: int,
      action: str,
      expected_output: str,
      burnin: int = 100,
      max_steps: int = 100,
      probe_output_filename: str = '',
  ):
    env = core.Environment(
        subject=subject,
        bug_triggering_input=bug_triggering_input,
        bug_trap=bug_trap,
        burnin=burnin,
        max_steps=max_steps,
        probe_output_filename=probe_output_filename,
        loglevel=0,
    )
    # TODO(etbarr): Test `execute_subject` and `update` methods.
    output = env.execute_subject()
    print(output)
    env.update(action=action)
    self.assertEqual(output, expected_output)


class LocaliserTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'test_a',
          'subject': SUBJECT_FILEPATH,
          'bug_triggering_input': '5',
          'bug_trap': BUG_TRAP,
      },
  )
  def test_generate_probes_random(
      self,
      subject: str,
      bug_triggering_input: str,
      bug_trap: int,
      burnin: int = 10,
      max_steps: int = 100,
      probe_output_filename: str = 'probe_output.txt',
  ):
    env = core.Environment(
        subject=subject,
        bug_triggering_input=bug_triggering_input,
        bug_trap=bug_trap,
        burnin=burnin,
        max_steps=max_steps,
        probe_output_filename=probe_output_filename,
        loglevel=0,
    )
    localiser = core.Localiser(env)
    localiser._generate_probes_random(env.state)  # pylint: disable=protected-access


if __name__ == '__main__':
  absltest.main()
