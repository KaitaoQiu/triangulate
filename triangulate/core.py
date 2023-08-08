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

"""Bug localization as a RL task."""

import abc
import ast
from collections.abc import Sequence
import dataclasses
import math
import tempfile
from typing import Any, IO

from absl import logging
from triangulate import ast_utils
from triangulate import instrumentation_utils
from triangulate import logging_utils
from triangulate import sampling_utils

PROBE_FUNCTION_NAME = instrumentation_utils.PROBE_FUNCTION_NAME
print_color = logging_utils.print_color
print_horizontal_line = logging_utils.print_horizontal_line

################################################################################
# Utilities
################################################################################


@dataclasses.dataclass
class CouldNotResolveIllegalStateExpressionError(Exception):
  """Illegal state expression could not be resolved for code and line number."""

  code: str
  lineno: int


def leading_whitespace_count(s: str) -> int:
  return len(s) - len(s.lstrip())


def prepend_line_numbers(s: str) -> str:
  lines = s.splitlines()
  line_count_width = len(str(len(lines)))
  lines_with_numbers = []
  for i, line in enumerate(lines):
    line_number = str(i).rjust(line_count_width)
    lines_with_numbers.append(f"{line_number}: {line}")
  return "\n".join(lines_with_numbers)


################################################################################
# Bug localization state
################################################################################


@dataclasses.dataclass
class Probe:
  """A probe statement for bug localization."""

  line_number: int
  statement: str


class State:
  """Bug localization state.

  Attributes:
    code: The source code being instrumented.
    code_lines: The source code lines in the current agent window.
    illegal_state_expression: The illegal state expression.
    focal_expression: The current expression under focus for bug localization.
    subject_with_probes: File descriptor of the subject program.
    probes: Probe statements, added by an agent.
  """

  def __init__(
      self,
      subject_with_probes: IO[str],
      bug_lineno: int | None = None,
      probes: Sequence[Probe] = (),
  ):
    self.code = subject_with_probes.read()
    self.code_lines = self.code.splitlines(keepends=True)
    if bug_lineno is not None:
      assert (
          0 <= bug_lineno < len(self.code_lines)
      ), "Bug line number out of bounds"

    illegal_state_expression = ast_utils.extract_illegal_state_expression(
        self.code, bug_lineno
    )
    if illegal_state_expression is None:
      raise CouldNotResolveIllegalStateExpressionError(self.code, bug_lineno)
    print_color("Illegal state expression resolved:", color="yellow")
    print(illegal_state_expression)
    self.set_illegal_state_expression(illegal_state_expression)
    focal_expression = self.illegal_state_expression
    self.set_focal_expression(focal_expression)

    self.subject_with_probes = subject_with_probes
    self.subject_with_probes.seek(0)
    self.probes = probes

  def set_illegal_state_expression(self, illegal_state_expression: str) -> None:
    try:
      compile(illegal_state_expression, "<string>", "eval")
    except SyntaxError as e:
      err_template = "Error: %s is an invalid Python expression."
      logging.error(err_template, illegal_state_expression)
      # TODO(etbarr) add when Python 3.11 is available within Google
      # e.add_note(err_template % expr)
      raise e
    self.illegal_state_expression = illegal_state_expression

  def set_focal_expression(self, focal_expression: str) -> None:
    try:
      compile(focal_expression, "<string>", "eval")
    except SyntaxError as e:
      err_template = "Error: %s is an invalid Python expression."
      logging.error(err_template, focal_expression)
      # TODO(etbarr) add when Python 3.11 is available within Google
      # e.add_note(err_template % expr)
      raise e
    self.focal_expression = focal_expression

  def get_illegal_state_expression_identifiers(self) -> Sequence[str]:
    """Return identifiers in the illegal state expression."""
    return ast_utils.extract_identifiers(self.illegal_state_expression)

  def __str__(self) -> str:
    # TODO(danielzheng): Implement a useful str representation.
    data = dict(
        subject_with_probes=self.subject_with_probes, code_lines=self.code_lines
    )
    return str(data)


# TODO(danielzheng): Add a real action type.
Action = Any


class Environment:
  """Bug localization environment.

  Attributes:
    state: The environment state.
    subject: The subject program filepath.
    subject_argv: Subject program arguments.
    subject_output: Set of outputs (stdout and stderr) from running the
      instrumented subject program.
    steps: The number of steps executed so far.
    max_burnin_steps: The maximum number of warmup steps to execute.
    max_steps: The maximum number of steps before termination.
    subject_with_probes: File descriptor of the subject program.
  """

  def __init__(
      self,
      subject: str,
      subject_argv: Sequence[str],
      bug_lineno: int,
      burnin_steps: int | None = None,
      max_steps: int | None = None,
  ):
    """Construct an environment instance."""
    self.subject = subject
    self.subject_argv = subject_argv
    self.subject_output = set()
    self.subject_with_probes = None
    self.steps = 0
    self.max_steps = max_steps if max_steps is not None else 100
    if burnin_steps:
      self.max_burnin_steps = math.ceil(burnin_steps * self.max_steps)
    else:
      self.max_burnin_steps = self.max_steps
    try:
      # pylint: disable=consider-using-with
      if logging.get_verbosity() == logging.DEBUG:
        self.subject_with_probes = tempfile.NamedTemporaryFile(
            mode="r+", delete=False
        )
        print(
            f"The subject with probes saved to {self.subject_with_probes.name}."
        )
      else:
        self.subject_with_probes = tempfile.NamedTemporaryFile(
            mode="r+", delete=True
        )
      # pylint: enable=consider-using-with
      with open(self.subject, "r", encoding="utf8") as f:
        data = f.read()
      self.subject_with_probes.write(data)
      self.subject_with_probes.flush()
      self.subject_with_probes.seek(0)
    except IOError as e:
      logging.error("Error: Unable to open file '%s'.", self.subject)
      raise IOError(
          "Unable to make a temporary copy of the subject program "
          f"{self.subject} to be instrumented with probes."
      ) from e
    self.state = State(self.subject_with_probes, bug_lineno)
    subject_output = self.execute_subject()
    self.subject_output.add(subject_output)

  def execute_subject(self) -> str:
    """Executes an instrumented version of the subject program.

    Returns:
      The subject's output, concatenating stdout and stderr.
    """
    assert self.subject_with_probes is not None
    self.subject_with_probes.seek(0)

    python_source = self.subject_with_probes.read()
    output = instrumentation_utils.run_with_instrumentation(
        python_source=python_source,
        argv=self.subject_argv,
    )

    # TODO(danielzheng): Do logging.
    print_color("Subject output:")
    print_horizontal_line()
    print(output)
    print_horizontal_line()
    return output

  def reward(self) -> int:
    """Returns reward for current state."""
    return 1

  def terminate(self) -> bool:
    """Returns True if environment execution should be terminated."""

    if self.steps >= self.max_steps:
      return True
    return False

  def update(self, action: Action) -> None:
    """Apply an action.

    Args:
      action: Action to apply to the environment.
    """
    match action:
      case "Placeholder":
        pass
      case _:
        pass
    self.steps += 1

    stdouterr = self.execute_subject()
    # Check that adding probes has not changed the buggy program's semantics
    # This check --- for whether we've seen the output during burnin ---
    # is an instance of the coupon collector's problem.
    if self.steps > self.max_burnin_steps:
      error_message = (
          "Error: probe insertion or execution changed program semantics."
      )
      if stdouterr not in self.subject_output:
        logging.exception(error_message)
        raise AssertionError(error_message)

    self.subject_output.add(stdouterr)
    # TODO(etbarr) Create and return a new state instance
    # Probe's write their output to a fresh file


################################################################################
# Bug localization agents
################################################################################


class Agent(abc.ABC):
  """RL agent base class for bug localization."""

  def __init__(self, env: Environment):
    self.env = env

  @abc.abstractmethod
  def pick_action(self, state: State, reward: int) -> Action:
    """Pick an action given the current state and reward.

    Args:
      state: Current state.
      reward: Reward for the last action.

    Returns:
      An action to apply to `state`.
    """

  def add_probes(self, state: State, probes: Sequence[Probe]) -> None:
    """Add probes to `state`.

    Args:
      state: Current state.
      probes: Sequence of `(line_number, probe_statement)` probes.
    """
    print_color("Adding probes:")
    for probe in probes:
      print(f"  {probe.line_number}: {probe.statement}")
    # TODO(danielzheng): Rewrite probe insertion using AST to avoid manual
    # indentation management.
    for probe in reversed(probes):
      offset = probe.line_number
      preceding_line = state.code_lines[offset]
      indentation = " " * leading_whitespace_count(preceding_line)
      probe_statement = indentation + probe.statement
      state.code_lines.insert(offset, probe_statement)
    state.subject_with_probes.seek(0)
    state.subject_with_probes.writelines(state.code_lines)

    # TODO(danielzheng): Do logging.
    print_color("New state with probes:")
    print_horizontal_line()
    state.subject_with_probes.seek(0)
    code = state.subject_with_probes.read()
    print(prepend_line_numbers(code))
    print_horizontal_line()


class Localiser(Agent):
  """Heuristic bug localization agent using code analysis."""

  def pick_action(self, state: State, reward: int) -> Action:
    """Picks action to apply to state."""
    del reward
    self.add_probes(state, self.generate_probes(state))

  def generate_probes(self, state: State) -> Sequence[Probe]:
    """Generate probes for the given state.

    To create each probe, this function must decide whether to query what.

    Args:
      state: The current state.

    Returns:
      Sequence of `(line_number, probe_statement)` probes.
    """
    return self._generate_probes_random(state)

  def _generate_probes_random(self, state: State) -> Sequence[Probe]:
    """Generate probes for the given state.

    Args:
      state: The current state.

    Returns:
      Sequence of `(line_number, probe_statement)` probes.
    """

    # TODO(danielzheng): Rewrite as methods on an AST wrapper
    state.subject_with_probes.seek(0)
    tree = ast.parse(state.subject_with_probes.read())
    all_insertion_line_numbers = ast_utils.get_insertion_points(tree)
    samples = sampling_utils.sample_zipfian(1, len(all_insertion_line_numbers))
    line_numbers = sampling_utils.sample_wo_replacement_uniform(
        samples[0], all_insertion_line_numbers
    )
    line_numbers.sort()

    # TODO(danielzheng): Use proper AST injection instead of string manipulation
    # and writing directly to files, to avoid malformed ASTs.
    probe_variables = state.get_illegal_state_expression_identifiers()
    probe_statement = instrumentation_utils.make_probe_call(probe_variables)
    probes = [(Probe(offset, probe_statement)) for offset in line_numbers]
    state.probes = probes
    return probes

  # TODO(etbarr): Build AST, reverse its edges and walk the tree from focal
  # expression to control expressions and defs
  # Ignore aliases for now.
  def _generate_probes_via_flow_analysis(self, state: State) -> Sequence[Probe]:
    """Use analysis techniques to generate probes for the given state."""
    del state
    raise NotImplementedError()


################################################################################
# Entry point
################################################################################


def run(
    subject: str,
    subject_argv: Sequence[str],
    bug_lineno: int,
    burnin_steps: int | None = None,
    max_steps: int | None = None,
):
  """Runs bug localization."""
  env = Environment(
      subject=subject,
      subject_argv=subject_argv,
      bug_lineno=bug_lineno,
      burnin_steps=burnin_steps,
      max_steps=max_steps,
  )
  localiser = Localiser(env)
  while not env.terminate():
    print_color(f"Step {env.steps}:", color="blue")
    env.update(localiser.pick_action(env.state, env.reward()))
  print_color(f"Done: {env.steps} steps performed.", color="blue")
