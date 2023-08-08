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

from collections.abc import Sequence
import os
import re

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

################################################################################
# Test cases
################################################################################


class EnvironmentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='test_quoter_index_0',
          subject='quoter.py',
          subject_argv=['--index', '0'],
          action='<placeholder>',
          expected_output="""\
Arguments: index = 0, seed = 0
Today's inspirational quote:
"It does not matter how slowly you go as long as you do not stop." - Confucius
""",
      ),
      dict(
          testcase_name='test_quoter_no_argv',
          subject='quoter.py',
          action='<placeholder>',
          expected_output=re.compile(
              'AssertionError: The first quote was not selected.'
          ),
      ),
  )
  def test_execute_and_update(
      self,
      subject: str,
      action: str,
      expected_output: str | re.Pattern[str],
      subject_argv: Sequence[str] = (),
      bug_lineno: int | None = None,
      burnin: int = 100,
      max_steps: int = 100,
      probe_output_filename: str = '',
  ):
    subject = os.path.join(TESTDATA_DIRECTORY, subject)
    env = core.Environment(
        subject=subject,
        subject_argv=(subject,) + tuple(subject_argv),
        bug_lineno=bug_lineno,
        burnin=burnin,
        max_steps=max_steps,
        probe_output_filename=probe_output_filename,
    )
    # TODO(etbarr): Test `execute_subject` and `update` methods.
    # NOTE(danielzheng): Change printing to be on command line.
    output = env.execute_subject()
    env.update(action=action)
    # If output is a regex, check regex match.
    if isinstance(expected_output, re.Pattern):
      self.assertRegex(output, expected_output)
    # Otherwise, check string equality.
    else:
      self.assertEqual(output, expected_output)


class LocaliserTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='test_a',
          subject='quoter.py',
      ),
  )
  def test_generate_probes_random(
      self,
      subject: str,
      subject_argv: Sequence[str] = (),
      bug_lineno: int | None = None,
      burnin: int = 10,
      max_steps: int = 100,
      probe_output_filename: str = 'probe_output.txt',
  ):
    subject = os.path.join(TESTDATA_DIRECTORY, subject)
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
