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

"""Tests for agents."""

from collections.abc import Sequence

from absl.testing import absltest
from absl.testing import parameterized

from triangulate import agents
from triangulate import core
from triangulate import environment
from triangulate import test_utils

TESTDATA_DIRECTORY = test_utils.TESTDATA_DIRECTORY


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
    subject = TESTDATA_DIRECTORY / subject
    env = environment.Environment(
        subject=str(subject),
        subject_argv=subject_argv,
        bug_lineno=bug_lineno,
        burnin_steps=burnin_steps,
        max_steps=max_steps,
    )
    random_probing = agents.RandomProbing()
    random_probing.generate_probes(env.state)


FUNCTION_CALL_ARGV = ['function_call.py', '0', '1', '2', '3']


class InspectVariablesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='function_call_bfs',
          subject='function_call.py',
          subject_argv=FUNCTION_CALL_ARGV,
          agent=agents.AgentEnum.BFS,
          expected_inspected_variable_names=(
              'd',
              'distance',
              'x1',
              'x2',
              'y1',
              'y2',
              'math',
              'x_diff_squared',
              'y_diff_squared',
              'x_diff',
              'x_diff',
              'y_diff',
              'y_diff',
              'x1',
              'x2',
              'y1',
              'y2',
          ),
      ),
      dict(
          testcase_name='function_call_dfs',
          subject='function_call.py',
          subject_argv=FUNCTION_CALL_ARGV,
          agent=agents.AgentEnum.DFS,
          expected_inspected_variable_names=(
              'd',
              'y2',
              'y1',
              'x2',
              'x1',
              'distance',
              'y_diff_squared',
              'y_diff',
              'y2',
              'y1',
              'y_diff',
              'x_diff_squared',
              'x_diff',
              'x2',
              'x1',
              'x_diff',
              'math',
          ),
      ),
  )
  def test_inspected_variables(
      self,
      subject: str,
      subject_argv: Sequence[str],
      agent: agents.AgentEnum,
      expected_inspected_variable_names: Sequence[str],
  ):
    subject = TESTDATA_DIRECTORY / subject
    result = core.run(
        subject=str(subject),
        subject_argv=subject_argv,
        agent=agent.make_agent(),
    )
    self.assertIsInstance(result, core.Result)
    assert isinstance(result, core.Result)  # For pytype.
    inspected_variables = result.final_state.inspected_variables
    inspected_variable_names = tuple(n.node.id for n in inspected_variables)
    self.assertSequenceEqual(
        inspected_variable_names,
        expected_inspected_variable_names,
    )


if __name__ == '__main__':
  absltest.main()
