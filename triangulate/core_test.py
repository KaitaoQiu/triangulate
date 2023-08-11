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
from typing import Type

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
      probes: core.Probes = (),
      subject_argv: Sequence[str] = (),
      bug_lineno: int | None = None,
      burnin_steps: int = 0,
      max_steps: int = 10,
  ):
    subject = os.path.join(TESTDATA_DIRECTORY, subject)
    env = core.Environment(
        subject=subject,
        subject_argv=(subject,) + tuple(subject_argv),
        bug_lineno=bug_lineno,
        burnin_steps=burnin_steps,
        max_steps=max_steps,
    )
    output = env.execute_subject()
    env.step(action=core.AddProbes(probes=probes))
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
    """Known unsupported cases for illegal state expression resolution."""
    subject = os.path.join(TESTDATA_DIRECTORY, subject)
    with self.assertRaises(core.CouldNotResolveIllegalStateExpressionError):
      _ = core.Environment(
          subject=subject,
          subject_argv=(subject,) + tuple(subject_argv),
          bug_lineno=None,
      )


class RandomProbingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='quoter',
          subject='quoter.py',
      ),
  )
  def test_generate_probes_random(
      self,
      subject: str,
      subject_argv: Sequence[str] = (),
      bug_lineno: int | None = None,
      burnin_steps: int = 10,
      max_steps: int = 100,
  ):
    subject = os.path.join(TESTDATA_DIRECTORY, subject)
    env = core.Environment(
        subject=subject,
        subject_argv=subject_argv,
        bug_lineno=bug_lineno,
        burnin_steps=burnin_steps,
        max_steps=max_steps,
    )
    random_probing = core.RandomProbing()
    random_probing.generate_probes(env.state)


class MainTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='quoter_no_exception_raised',
          subject='quoter.py',
          subject_argv=['quoter.py', '--index', '0'],
          expected_result=core.NoExceptionRaised,
      ),
      dict(
          testcase_name='quoter_exception_raised',
          subject='quoter.py',
          subject_argv=['quoter.py', '--index', '1'],
          expected_result=core.Result,
      ),
  )
  def test_run(
      self,
      subject: str,
      expected_result: Type[core.ResultOrError] | core.ResultOrError,
      subject_argv: Sequence[str] = (),
      burnin_steps: int = 10,
      max_steps: int = 100,
  ):
    subject = os.path.join(TESTDATA_DIRECTORY, subject)
    result = core.run(
        subject=subject,
        subject_argv=subject_argv,
        agent=core.RandomProbing(),
        burnin_steps=burnin_steps,
        max_steps=max_steps,
    )
    if isinstance(expected_result, type):
      self.assertEqual(type(result), expected_result)
    else:
      self.assertEqual(result, expected_result)


if __name__ == '__main__':
  absltest.main()
