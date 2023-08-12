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

"""Tests for environment."""

from collections.abc import Sequence
import re

from absl.testing import absltest
from absl.testing import parameterized

from triangulate import core
from triangulate import environment
from triangulate import exceptions
from triangulate import test_utils

TESTDATA_DIRECTORY = test_utils.TESTDATA_DIRECTORY


class EnvironmentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='test_quoter_index_0',
          subject='quoter.py',
          subject_argv=['--index', '0'],
          expected_output="""\
Arguments: index = 0, seed = 0
Today's inspirational quote:
"It does not matter how slowly you go as long as you do not stop." - Confucius
""",
      ),
      dict(
          testcase_name='test_quoter_no_argv',
          subject='quoter.py',
          expected_output=re.compile(
              'AssertionError: The first quote was not selected.'
          ),
      ),
  )
  def test_execute_and_update(
      self,
      subject: str,
      expected_output: str | re.Pattern[str],
      probes: environment.Probes = (),
      subject_argv: Sequence[str] = (),
      bug_lineno: int | None = None,
      burnin_steps: int = 0,
      max_steps: int = 10,
  ):
    subject = TESTDATA_DIRECTORY / subject
    env = core.Environment(
        subject=str(subject),
        subject_argv=(subject,) + tuple(subject_argv),
        bug_lineno=bug_lineno,
        burnin_steps=burnin_steps,
        max_steps=max_steps,
    )
    output = env.execute_subject()
    env.step(action=environment.AddProbes(probes=probes))
    # If output is a regex, check regex match.
    if isinstance(expected_output, re.Pattern):
      self.assertRegex(output, expected_output)
    # Otherwise, check string equality.
    else:
      self.assertEqual(output, expected_output)

  @parameterized.named_parameters(
      dict(
          testcase_name='test_multifile_main',
          subject='multifile_example/main.py',
      ),
  )
  def test_could_not_resolve_illegal_state_expression(
      self,
      subject: str,
      subject_argv: Sequence[str] = (),
  ):
    """Known unsupported cases for illegal state expression identification."""
    subject = TESTDATA_DIRECTORY / subject
    with self.assertRaises(
        exceptions.CouldNotIdentifyIllegalStateExpressionError
    ):
      _ = core.Environment(
          subject=str(subject),
          subject_argv=(subject,) + tuple(subject_argv),
          bug_lineno=None,
      )


if __name__ == '__main__':
  absltest.main()
