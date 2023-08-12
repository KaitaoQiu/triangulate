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

from absl.testing import absltest
from absl.testing import parameterized

from triangulate import agents
from triangulate import core
from triangulate import exceptions
from triangulate import test_utils

TESTDATA_DIRECTORY = test_utils.TESTDATA_DIRECTORY


class MainTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='quoter_no_exception_raised',
          subject='quoter.py',
          subject_argv=['quoter.py', '--index', '0'],
          expected_result=exceptions.SubjectProgramNoExceptionRaisedError,
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
      expected_result: type[core.ResultOrError] | core.ResultOrError,
      subject_argv: Sequence[str] = (),
      burnin_steps: int = 10,
      max_steps: int = 100,
  ):
    subject = TESTDATA_DIRECTORY / subject
    result = core.run(
        subject=str(subject),
        subject_argv=subject_argv,
        agent=agents.RandomProbing(),
        burnin_steps=burnin_steps,
        max_steps=max_steps,
    )
    if isinstance(expected_result, type):
      self.assertEqual(type(result), expected_result)
    else:
      self.assertEqual(result, expected_result)


if __name__ == '__main__':
  absltest.main()
