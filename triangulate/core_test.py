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


TESTDATA_DIRECTORY = os.path.join(
    absltest.get_default_test_srcdir(),
    'triangulate/testdata',
)
SUBJECT_FILENAME = 'quoter.py'
SUBJECT_FILEPATH = os.path.join(TESTDATA_DIRECTORY, SUBJECT_FILENAME)
with open(SUBJECT_FILEPATH, 'r', encoding='utf8') as f:
  SUBJECT_CONTENT = f.read()

################################################################################
# Test cases
################################################################################


class EnvironmentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='test_a',
          subject=SUBJECT_FILEPATH,
          subject_argv='42',
          action='<placeholder>',
          expected_output="""\
Today's inspirational quote:
"Believe you can and you're halfway there." - Theodore Roosevelt
""",
      ),
      dict(
          testcase_name='test_b',
          subject=SUBJECT_FILEPATH,
          subject_argv='42',
          action='<placeholder>',
          expected_output="""\
Today's inspirational quote:
"Believe you can and you're halfway there." - Theodore Roosevelt
""",
      ),
  )
  def test_execute_and_update(
      self,
      subject: str,
      subject_argv: str,
      action: str,
      expected_output: str,
      bug_lineno: int | None = None,
      burnin: int = 100,
      max_steps: int = 100,
      probe_output_filename: str = '',
  ):
    env = core.Environment(
        subject=subject,
        subject_argv=subject_argv,
        bug_lineno=bug_lineno,
        burnin=burnin,
        max_steps=max_steps,
        probe_output_filename=probe_output_filename,
    )
    # TODO(etbarr): Test `execute_subject` and `update` methods.
    output = env.execute_subject()
    print(output)
    env.update(action=action)
    self.assertEqual(output, expected_output)


class LocaliserTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='test_a',
          subject=SUBJECT_FILEPATH,
          subject_argv='5',
      ),
  )
  def test_generate_probes_random(
      self,
      subject: str,
      subject_argv: str,
      bug_lineno: int | None = None,
      burnin: int = 10,
      max_steps: int = 100,
      probe_output_filename: str = 'probe_output.txt',
  ):
    env = core.Environment(
        subject=subject,
        subject_argv=subject_argv,
        bug_lineno=bug_lineno,
        burnin=burnin,
        max_steps=max_steps,
        probe_output_filename=probe_output_filename,
    )
    localiser = core.Localiser(env)
    localiser._generate_probes_random(env.state)  # pylint: disable=protected-access


if __name__ == '__main__':
  absltest.main()
