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


Probes = tuple[Probe, ...]


@dataclasses.dataclass
class State:
  """Bug localization state.

  Attributes:
    code: The source code being instrumented.
    code_lines: The source code lines in the current agent window.
    illegal_state_expression: The illegal state expression.
    focal_expression: The current expression under focus for bug localization.
    probes: Probe statements, added by an agent.
  """

  code: str
  bug_lineno: int | None = None
  probes: Probes = ()

  code_lines: Sequence[str] = dataclasses.field(init=False)
  illegal_state_expression: str = dataclasses.field(init=False)
  focal_expression: str = dataclasses.field(init=False)

  def __post_init__(self):
    self.code_lines = tuple(self.code.splitlines(keepends=True))

    if self.bug_lineno is not None:
      if not 0 <= self.bug_lineno < len(self.code_lines):
        raise ValueError("Bug line number out of bounds")

    illegal_state_expression = ast_utils.extract_illegal_state_expression(
        self.code, self.bug_lineno
    )
    if illegal_state_expression is None:
      raise CouldNotResolveIllegalStateExpressionError(
          self.code, self.bug_lineno
      )

    print_color("Illegal state expression resolved:", color="yellow")
    print(illegal_state_expression)
    self.set_illegal_state_expression(illegal_state_expression)
    focal_expression = self.illegal_state_expression
    self.set_focal_expression(focal_expression)

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

  def render_code(self) -> str:
    """Renders the code with probes inserted."""
    # TODO(danielzheng): Rewrite probe insertion using AST to avoid manual
    # indentation management.
    new_code_lines = list(self.code_lines)
    for probe in reversed(self.probes):
      offset = probe.line_number
      preceding_line = self.code_lines[offset]
      indentation = " " * leading_whitespace_count(preceding_line)
      probe_statement = indentation + probe.statement
      new_code_lines.insert(offset, probe_statement)
    return "".join(new_code_lines)

  def __str__(self) -> str:
    # TODO(danielzheng): Implement a useful str representation.
    data = dict(code_lines=self.code_lines)
    return str(data)


################################################################################
# Bug localization actions
################################################################################


class Action(abc.ABC):
  """Bug localization action."""

  @abc.abstractmethod
  def update(self, state: State) -> State:
    """Applies `self` to `state` to get a new state."""


class Halt(Action):
  """A no-op halt action that does nothing.

  The `run` function recognizes this action and terminates the RL loop.
  """

  def update(self, state: State) -> State:
    return state


@dataclasses.dataclass
class AddProbes(Action):
  """Add probes to `state`.

  Attributes:
    probes: Sequence of `(line_number, probe_statement)` probes.
  """

  probes: Probes

  def update(self, state: State) -> State:
    """Updates state with new probes."""
    print_color("Adding probes:")
    for probe in self.probes:
      print(f"  {probe.line_number}: {probe.statement}")

    # TODO(danielzheng): Optimize this.
    new_probes = tuple(
        sorted(state.probes + self.probes, key=lambda probe: probe.line_number)
    )
    new_state = dataclasses.replace(state, probes=new_probes)

    # TODO(danielzheng): Do logging.
    print_color("New state with probes:")
    print_horizontal_line()
    print(prepend_line_numbers(new_state.render_code()))
    print_horizontal_line()
    return new_state


################################################################################
# Bug localization agents
################################################################################


Reward = int


class Agent(abc.ABC):
  """Bug localization agent."""

  @abc.abstractmethod
  def pick_action(self, state: State, reward: Reward) -> Action:
    """Pick an action given the current state and reward.

    Args:
      state: Current state.
      reward: Reward for the last action.

    Returns:
      An action to apply to `state`.
    """


class Localiser(Agent):
  """Heuristic bug localization agent using code analysis."""

  def pick_action(self, state: State, reward: Reward) -> Action:
    """Picks action to apply to state."""
    del reward
    probes = self.generate_probes(state)
    # If no probes are generated, then halt.
    if not probes:
      return Halt()
    return AddProbes(probes)

  def generate_probes(self, state: State) -> Probes:
    """Generate probes for the given state.

    To create each probe, this function must decide whether to query what.

    Args:
      state: The current state.

    Returns:
      Sequence of `(line_number, probe_statement)` probes.
    """
    return self._generate_probes_random(state)

  def _generate_probes_random(self, state: State) -> Probes:
    """Generate probes for the given state.

    Args:
      state: The current state.

    Returns:
      Sequence of `(line_number, probe_statement)` probes.
    """

    tree = ast.parse(state.code)
    # Get all valid insertion line numbers.
    all_insertion_line_numbers = ast_utils.get_insertion_points(tree)
    # Remove line numbers that already have probes.
    valid_insertion_line_numbers = all_insertion_line_numbers - set(
        probe.line_number for probe in state.probes
    )
    if not valid_insertion_line_numbers:
      return ()
    # Sample a single new probe.
    samples = sampling_utils.sample_zipfian(
        num_samples=1, zipf_param=len(all_insertion_line_numbers)
    )
    line_numbers = sampling_utils.sample_wo_replacement_uniform(
        samples[0], tuple(valid_insertion_line_numbers)
    )
    line_numbers.sort()

    probe_variables = state.get_illegal_state_expression_identifiers()
    probe_statement = instrumentation_utils.make_probe_call(probe_variables)
    probes = tuple((Probe(offset, probe_statement)) for offset in line_numbers)
    return probes

  # TODO(etbarr): Build AST, reverse its edges and walk the tree from focal
  # expression to control expressions and defs
  # Ignore aliases for now.
  def _generate_probes_via_flow_analysis(self, state: State) -> Probes:
    """Use analysis techniques to generate probes for the given state."""
    del state
    raise NotImplementedError()


################################################################################
# Bug localization environment
################################################################################


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
        subject_code = f.read()
    except IOError as e:
      logging.error("Error: Unable to open file '%s'.", self.subject)
      raise IOError(
          "Unable to make a temporary copy of the subject program "
          f"{self.subject} to be instrumented with probes."
      ) from e
    self.state = State(code=subject_code, bug_lineno=bug_lineno)
    print("Initial state:")
    print(self.state)
    subject_output = self.execute_subject()
    self.subject_output.add(subject_output)

  def execute_subject(self) -> str:
    """Executes an instrumented version of the subject program.

    Returns:
      The output of the instrumented subject program, concatenating stdout and
      stderr.
    """
    python_source = self.state.render_code()
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

  def step(self, action: Action) -> None:
    """Perform one step by applying an action.

    Args:
      action: Action to apply to the environment.
    """
    self.state = action.update(self.state)
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

  def reward(self) -> Reward:
    """Returns reward for current state."""
    return 1

  def terminate(self) -> bool:
    """Returns True if environment execution should be terminated."""
    if self.steps >= self.max_steps:
      return True
    return False


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
  agent = Localiser()
  while not env.terminate():
    print_color(f"Step {env.steps}:", color="blue")
    action = agent.pick_action(env.state, env.reward())
    if isinstance(action, Halt):
      print_color("Stopping due to halt action.", color="blue")
      break
    env.step(action)
  print_color(f"Done: {env.steps} steps performed.", color="blue")
