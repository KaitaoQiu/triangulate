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
from collections.abc import Sequence, Set
import contextlib
import dataclasses
import enum
import io
import math
import queue
import runpy
import sys
import tempfile
import traceback
import types
from typing import Callable, Generic, TypeAlias, TypeVar

from absl import logging
import ordered_set
from python_graphs import program_graph
from python_graphs import program_graph_dataclasses as pb
from rich.syntax import Syntax

from triangulate import ast_utils
from triangulate import instrumentation_utils
from triangulate import logging_utils
from triangulate import sampling_utils

OrderedSet = ordered_set.OrderedSet
ProgramGraph = program_graph.ProgramGraph
ProgramGraphNode = program_graph.ProgramGraphNode

PROBE_FUNCTION_NAME = instrumentation_utils.PROBE_FUNCTION_NAME

CONSOLE = logging_utils.CONSOLE
rprint = CONSOLE.print
print_horizontal_line = logging_utils.print_horizontal_line
print_panel = logging_utils.print_panel

################################################################################
# Utilities
################################################################################


class BugLocalizationException(Exception):
  """A bug localization exception."""


@dataclasses.dataclass
class NoExceptionRaised(BugLocalizationException):
  """Subject program does not raise an exception."""

  subject: str
  subject_argv: Sequence[str]


@dataclasses.dataclass
class LineNumberOutOfBounds(BugLocalizationException):
  """Line number is out of bounds for program."""

  code: str
  lineno: int

  def __str__(self) -> str:
    return (
        f"Line number {self.lineno} is out of bounds for code:\n"
        f"{prepend_line_numbers(self.code)}"
    )


@dataclasses.dataclass
class CouldNotResolveIllegalStateExpressionError(BugLocalizationException):
  """Illegal state expression could not be resolved for code and line number."""

  code: str
  lineno: int


@dataclasses.dataclass
class VariableAlreadyInspected(BugLocalizationException):
  """Cannot inspect variable that has already been inspected."""

  variable: str
  # inspected_variables: Set[str]
  # candidate_variables_to_inspect: Set[str]
  inspected_variables: OrderedSet[ProgramGraphNode]
  candidate_variables_to_inspect: OrderedSet[ProgramGraphNode]


@dataclasses.dataclass
class VariableNotInspectionCandidate(BugLocalizationException):
  """Cannot inspect variable that is not a candidate for inspection."""

  variable: str
  # inspected_variables: Set[str]
  # candidate_variables_to_inspect: Set[str]
  inspected_variables: OrderedSet[ProgramGraphNode]
  candidate_variables_to_inspect: OrderedSet[ProgramGraphNode]


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
    inspected_variables: The variables inspected by an agent.
    candidate_variables_to_inspect: The variables that an agent can inspect.
    probes: Probe statements, added by an agent.
  """

  code: str
  bug_lineno: int | None = None
  # TODO(danielzheng): Need to store AST information as well.
  # - Inspected variable values also depends on code location.
  # - Variable shadowing also needs to be handled.
  # TODO(danielzheng): Generalize to properties, e.g. `sys.argv`.
  # - Generalization of "variables" - this represents a `probe` statement like
  #   `print` in a debugger, along the execution path towards the exception.
  # inspected_variables: Set[str] = frozenset()
  # candidate_variables_to_inspect: Set[str] | None = None
  inspected_variables: OrderedSet[ProgramGraphNode] = OrderedSet()
  candidate_variables_to_inspect: OrderedSet[ProgramGraphNode] | None = None
  probes: Probes = ()

  code_lines: Sequence[str] | None = None
  illegal_state_expression: str | None = None

  program_graph: ProgramGraph | None = None
  illegal_state_expression_node: ProgramGraphNode | None = None

  def __post_init__(self):
    if self.code_lines is None:
      self.code_lines = tuple(self.code.splitlines(keepends=True))

    if self.bug_lineno is not None:
      if not 0 <= self.bug_lineno < len(self.code_lines):
        raise LineNumberOutOfBounds(self.code, self.bug_lineno)

    # TODO(danielzheng): Make this condition precise.
    if self.illegal_state_expression is None:
      illegal_state_expression = ast_utils.extract_illegal_state_expression(
          self.code, self.bug_lineno
      )
      if illegal_state_expression is None:
        raise CouldNotResolveIllegalStateExpressionError(
            self.code, self.bug_lineno
        )

      rprint("Illegal state expression identified:", style="bold yellow")
      rprint(illegal_state_expression, style="yellow", highlight=False)
      self.illegal_state_expression = illegal_state_expression

    if self.program_graph is None:
      self.program_graph = program_graph.get_program_graph(self.code)

    if self.illegal_state_expression_node is None:
      self.illegal_state_expression_node = (
          self.program_graph.get_node_by_source(self.illegal_state_expression)
      )

    if self.candidate_variables_to_inspect is None:
      self.candidate_variables_to_inspect = OrderedSet(
          self.get_illegal_state_expression_identifiers()
      )

  def get_illegal_state_expression_identifiers(
      self,
  ) -> OrderedSet[ProgramGraphNode]:
    """Returns all variable identifiers in the illegal state expression."""
    return OrderedSet(
        ast_utils.get_ast_descendents_of_type(
            self.program_graph, self.illegal_state_expression_node, ast.Name
        )
    )

  def inspect_variable(self, variable: ProgramGraphNode) -> "State":
    rprint("Inspecting variable:", style=logging_utils.ACTION_STYLE)
    rprint(repr(variable.node.id), variable)
    if variable not in self.candidate_variables_to_inspect:
      raise VariableNotInspectionCandidate(
          variable,
          self.inspected_variables,
          self.candidate_variables_to_inspect,
      )

    if variable in self.inspected_variables:
      raise VariableAlreadyInspected(
          variable,
          self.inspected_variables,
          self.candidate_variables_to_inspect,
      )

    new_inspected_variables = self.inspected_variables | {variable}
    new_candidate_variables_to_inspect = self.candidate_variables_to_inspect - {
        variable
    }

    # Update candidate variables to inspect.
    parent_node: ProgramGraphNode = self.program_graph.parent(variable)
    if parent_node.ast_type == "Call":
      rprint("parent_node", style="bold magenta")
      print(self.program_graph.dump_tree(parent_node))
      function_definition_nodes = self.program_graph.outgoing_neighbors(
          parent_node, edge_type=pb.EdgeType.CALLS
      )
      for function_definition_node in function_definition_nodes:
        rprint("Function definition node:", style="yellow")
        print(self.program_graph.dump_tree(function_definition_node))
        return_nodes = ast_utils.get_ast_descendents_of_type(
            self.program_graph, function_definition_node, ast.Return
        )
        for return_node in return_nodes:
          rprint("Return node:", style="yellow")
          print(self.program_graph.dump_tree(return_node))
          return_value_nodes = tuple(self.program_graph.children(return_node))
          rprint(
              f"Return value nodes {len(return_value_nodes)}:", style="yellow"
          )
          for return_value_node in return_value_nodes:
            rprint("Return value node:", style="yellow")
            print(self.program_graph.dump_tree(return_value_node))
            # TODO(danielzheng): Ignore module and attribute name identifiers.
            return_value_identifiers = ast_utils.get_ast_descendents_of_type(
                self.program_graph, return_value_node, ast.Name
            )
            new_candidate_variables_to_inspect.update(return_value_identifiers)

    definition_nodes = self.program_graph.outgoing_neighbors(
        variable, edge_type=pb.EdgeType.LAST_WRITE
    )
    for definition_node in definition_nodes:
      rprint("Definition node:", style="yellow")
      print(self.program_graph.dump_tree(definition_node))
      operand_nodes = self.program_graph.outgoing_neighbors(
          definition_node, edge_type=pb.EdgeType.COMPUTED_FROM
      )
      for operand_node in operand_nodes:
        rprint(f"Operand node for {definition_node}:", style="yellow")
        print(self.program_graph.dump_tree(operand_node))
      new_candidate_variables_to_inspect.update(operand_nodes)

    # TODO(danielzheng): Update illegal state expression with newly inspected
    # variables. Consider adding an `IllegalStateExpression` dataclass.
    return dataclasses.replace(
        self,
        inspected_variables=new_inspected_variables,
        candidate_variables_to_inspect=new_candidate_variables_to_inspect,
    )

  def inspect_variables(self, variables: Sequence[ProgramGraphNode]) -> "State":
    result = self
    for variable in variables:
      result = result.inspect_variable(variable)
    return result

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
    rprint("Adding probes:", style=logging_utils.ACTION_STYLE)
    for probe in self.probes:
      rprint(f"  {probe.line_number}: {probe.statement}")

    new_probes = tuple(
        sorted(state.probes + self.probes, key=lambda probe: probe.line_number)
    )
    new_state = dataclasses.replace(state, probes=new_probes)

    # Print rendered code.
    rendered_code = new_state.render_code()
    print_panel(
        Syntax(
            rendered_code,
            lexer="python",
            theme="ansi_light",
            line_numbers=True,
            start_line=0,
        ),
        title="New state with probes",
    )
    return new_state


T = TypeVar("T")


class VariableInspectionStrategy(Generic[T], abc.ABC):
  """Strategy for inspecting variables."""

  @abc.abstractmethod
  def select_variables_to_inspect(self, state: State) -> Sequence[T]:
    """Returns the next candidate variables to inspect."""


class BruteForceSearchStrategy(VariableInspectionStrategy, Generic[T], abc.ABC):
  """Brute force search, i.e. uninformed search."""

  frontier: queue.Queue
  visited: OrderedSet[T]

  def __init__(self):
    frontier_type = self.frontier_type()
    self.frontier = frontier_type()
    self.frontier_added = OrderedSet()
    self.visited = OrderedSet()

  @classmethod
  @abc.abstractmethod
  def frontier_type(cls) -> type[queue.Queue]:
    """Returns the frontier type for this search strategy."""

  def update_frontier(self, variables: Set[T]):
    for variable in variables:
      if variable in self.frontier_added:
        # TODO(danielzheng): Use debug logging below.
        # rprint(f"Skip adding to frontier: {(variable, variable.node.id)}")
        continue
      rprint(f"Adding to frontier: {(variable, variable.node.id)}")
      self.frontier.put(variable)
      self.frontier_added.add(variable)
    # TODO(danielzheng): Use debug logging below.
    # rprint("self.frontier.qsize", self.frontier.qsize())
    # rprint("self.visited", len(self.visited))

  def pop_frontier(self) -> T:
    value = self.frontier.get()
    self.visited.add(value)
    return value

  def select_variables_to_inspect(self, state: State) -> Sequence[T]:
    self.update_frontier(state.candidate_variables_to_inspect)
    if self.frontier.empty():
      return ()
    next_variable_to_visit = self.pop_frontier()
    # Explore one variable at a time.
    return (next_variable_to_visit,)


class BreadthFirstStrategy(BruteForceSearchStrategy):
  """Breadth first search strategy."""

  @classmethod
  def frontier_type(cls) -> type[queue.Queue]:
    return queue.Queue


class DepthFirstStrategy(BruteForceSearchStrategy):
  """Depth first search strategy."""

  @classmethod
  def frontier_type(cls) -> type[queue.Queue]:
    return queue.LifoQueue


@dataclasses.dataclass
class InspectVariable(Action):
  """Update state by inspecting a variable in the frontier."""

  variables_to_inspect: Sequence[ProgramGraphNode]

  def update(self, state: State) -> State:
    """Updates state by inspecting variables."""
    return state.inspect_variables(self.variables_to_inspect)


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


@dataclasses.dataclass
class Replayer(Agent):
  """Agent that executes a predefined sequence of actions."""

  actions: Sequence[Action]
  action_index: int = 0

  def pick_action(self, state: State, reward: Reward) -> Action:
    del state, reward  # Unused.
    if self.action_index == len(self.actions):
      return Halt()
    action = self.actions[self.action_index]
    self.action_index += 1
    return action


# TODO(danielzheng): Figure out how to make probes useful within agents.
# May need to update illegal state expression representation with probe info.
class ProbingAgent(Agent, abc.ABC):
  """Agent that inserts probes at random locations."""

  def pick_action(self, state: State, reward: Reward) -> Action:
    """Picks action to apply to state."""
    del reward
    probes = self.generate_probes(state)
    # If no probes are generated, then halt.
    if not probes:
      return Halt()
    return AddProbes(probes)

  @abc.abstractmethod
  def generate_probes(self, state: State) -> Probes:
    """Generate probes for the given state.

    Args:
      state: The current state.

    Returns:
      Sequence of `(line_number, probe_statement)` probes.
    """


class RandomProbing(ProbingAgent):
  """Agent that inserts probes at random locations."""

  def generate_probes(self, state: State) -> Probes:
    """Generate probes for the given state at a random valid location."""
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
    # Make probe statement, for inspecting the values of all variable in the
    # illegal state expression at a given line number.
    probe_variables = sorted(
        n.node.id for n in state.get_illegal_state_expression_identifiers()
    )
    probe_statement = instrumentation_utils.make_probe_call(probe_variables)
    return tuple((Probe(offset, probe_statement)) for offset in line_numbers)


class FlowBasedProbing(ProbingAgent):
  """Agent that inserts probes based on control flow analysis."""

  def generate_probes(self, state: State) -> Probes:
    """Generate probes for the given state via analysis techniques."""
    del state  # Unused.
    # TODO(etbarr): Build AST, reverse its edges and walk the tree from focal
    # expression to control expressions and definitions. Ignore aliases for now.
    raise NotImplementedError()


SearchStrategy = TypeVar("SearchStrategy", bound=BruteForceSearchStrategy)


@dataclasses.dataclass
class SearchAgent(Agent, Generic[SearchStrategy]):
  """Agent that inspects variables via brute force search."""

  strategy_type: type[SearchStrategy]
  strategy: SearchStrategy = dataclasses.field(init=False)

  def __post_init__(self):
    self.strategy = self.strategy_type()

  def pick_action(self, state: State, reward: Reward) -> Action:
    """Inspect variables based on a search strategy."""
    del reward  # Unused.
    variables = self.strategy.select_variables_to_inspect(state)
    # If there are no remaining variables to inspect, halt.
    if not variables:
      visited_names = tuple(n.node.id for n in self.strategy.visited)
      rprint(f"Inspected all variables: {visited_names}")
      assert set(state.inspected_variables) == self.strategy.visited
      return Halt()
    return InspectVariable(variables)

  @property
  def visited_variables(self) -> Set[str]:
    return self.strategy.visited


def make_search_agent_factory(
    strategy_type: type[SearchStrategy],
) -> Callable[[], Agent]:
  """Returns a factory function that creates a search agent with fresh state."""
  return lambda: SearchAgent(strategy_type=strategy_type)


make_breadth_first_search = make_search_agent_factory(BreadthFirstStrategy)
make_depth_first_search = make_search_agent_factory(DepthFirstStrategy)


class AgentEnum(enum.Enum):
  """Enumeration of bug localization agents.

  Enum values are `Agent` factory functions: `Callable[[], Agent]`.
  """

  RANDOM_PROBING = enum.auto()
  FLOW_BASED_PROBING = enum.auto()
  BFS = enum.auto()
  DFS = enum.auto()

  def make_agent(self) -> Agent:
    """Makes an agent."""
    match self:
      case self.RANDOM_PROBING:
        return RandomProbing()
      case self.FLOW_BASED_PROBING:
        return FlowBasedProbing()
      case self.BFS:
        return make_breadth_first_search()
      case self.DFS:
        return make_depth_first_search()
      case _:
        raise ValueError(f"Unknown agent type: {self}")


################################################################################
# Bug localization environment
################################################################################


class Environment:
  """Bug localization environment.

  Attributes:
    state: The environment state.
    subject: The subject program filepath.
    subject_argv: Subject program arguments, as represented like `sys.argv` when
      executing a Python program. `subject_argv[0]` should be the program name.
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
      bug_lineno: int | None = None,
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
        rprint(
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
    subject_output = self.execute_subject(print_output=False)
    self.subject_output.add(subject_output)

  def execute_subject(self, print_output: bool = True) -> str:
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

    if print_output:
      print_panel(output.removesuffix("\n"), title="Subject output")
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
# Entry points
################################################################################


@dataclasses.dataclass
class Result:
  """A bug localization result.

  TODO(danielzheng): Add attributes to `State` to make this more informative.
  It should be easy to access the final "predicted bug location" as a property.

  Attributes:
    final_state: The final bug localization state.
    localization_steps: The number of localization steps performed.
  """

  final_state: State
  localization_steps: int


# Bug localization result: either a concrete result or an exception.
ResultOrError: TypeAlias = Result | BugLocalizationException


# TODO(danielzheng): Fix this to take `illegal_state_expression` instead of
# `bug_lineno`. `bug_lineno` is imprecise and insufficient.
def run_with_bug_lineno(
    subject: str,
    subject_argv: Sequence[str],
    agent: Agent,
    bug_lineno: int,
    burnin_steps: int | None = None,
    max_steps: int | None = None,
) -> Result:
  """Runs bug localization with a given bug location.

  Args:
    subject: The subject program filepath.
    subject_argv: Subject program arguments, as represented like `sys.argv` when
      executing a Python program. `subject_argv[0]` should be the program name.
    agent: The agent for bug localization.
    bug_lineno: The line number of the bug in `subject`.
    burnin_steps: The maximum number of warmup steps to execute.
    max_steps: The maximum number of steps before termination.

  Returns:
    Bug localization result, containing the final predicted bug location.
  """
  env = Environment(
      subject=subject,
      subject_argv=subject_argv,
      bug_lineno=bug_lineno,
      burnin_steps=burnin_steps,
      max_steps=max_steps,
  )
  while not env.terminate():
    CONSOLE.rule(title=f"[bold white]Step {env.steps}")
    action = agent.pick_action(env.state, env.reward())
    if isinstance(action, Halt):
      rprint("Stopping due to halt action.", style="bold blue")
      break
    env.step(action)
  rprint(f"Done: {env.steps} steps performed.", style="bold blue")
  return Result(env.state, env.steps)


# The result type of `sys.exc_info()`.
ExcInfo: TypeAlias = tuple[
    type[BaseException], BaseException, types.TracebackType
]


def run_from_exception(
    subject: str,
    subject_argv: Sequence[str],
    agent: Agent,
    exc_info: ExcInfo,
    burnin_steps: int | None = None,
    max_steps: int | None = None,
) -> ResultOrError:
  """Runs bug localization for the given exception info.

  Uses `exc_info` to determine the illegal state expression.

  Args:
    subject: The subject program filepath.
    subject_argv: Subject program arguments, as represented like `sys.argv` when
      executing a Python program. `subject_argv[0]` should be the program name.
    agent: The agent for bug localization.
    exc_info: Exception information as returned by `sys.exc_info()`, including
      the exception information and traceback.
    burnin_steps: The maximum number of warmup steps to execute.
    max_steps: The maximum number of steps before termination.

  Returns:
    Bug localization result: `CouldNotResolveIllegalStateExpressionError` if the
    illegal state expression could not be resolved from the exception, or the
    result of `run_with_bug_lineno` otherwise.
  """
  rprint("Exception caught:", style="bold yellow")
  _, exc_value, tb = exc_info
  if exc_value is None:
    raise NoExceptionRaised(subject=subject, subject_argv=subject_argv)
  traceback.print_tb(tb)
  tb_info = traceback.extract_tb(tb)
  exc_last_frame = tb_info[-1]
  exc_lineno = exc_last_frame.lineno

  try:
    return run_with_bug_lineno(
        subject=subject,
        subject_argv=subject_argv,
        agent=agent,
        bug_lineno=exc_lineno,
        burnin_steps=burnin_steps,
        max_steps=max_steps,
    )
  except CouldNotResolveIllegalStateExpressionError as e:
    rprint(
        "Could not resolve illegal state expression from exception:",
        style="bold red",
    )
    traceback.print_exception(exc_value, limit=-1)
    return e


def run(
    subject: str,
    subject_argv: Sequence[str],
    agent: Agent,
    burnin_steps: int | None = None,
    max_steps: int | None = None,
) -> ResultOrError:
  """Runs bug localization for the given subject program and arguments.

  Executes the subject program with arguments. If an exception is thrown, the
  exception location is used as the bug location.

  Args:
    subject: The subject program filepath.
    subject_argv: Subject program arguments, as represented like `sys.argv` when
      executing a Python program. `subject_argv[0]` should be the program name.
    agent: The agent for bug localization.
    burnin_steps: The maximum number of warmup steps to execute.
    max_steps: The maximum number of steps before termination.

  Returns:
    Bug localization result: `NoExceptionRaised` if no exception was thrown by
    the subject program, or the result of `run_from_exception` otherwise.
  """
  # Rewrite `sys.argv` to subject program's argv.
  sys.argv = subject_argv

  buffer = io.StringIO()
  exc_info = None
  try:
    rprint(rf"[bold blue]Running:[/] {subject}")
    with (
        contextlib.redirect_stdout(buffer),
        contextlib.redirect_stderr(buffer),
    ):
      runpy.run_path(subject, run_name="__main__")
  except Exception:  # pylint: disable=broad-except
    exc_info = sys.exc_info()
  finally:
    print_panel(buffer.getvalue().removesuffix("\n"), title="Subject output")

  if exc_info is not None:
    return run_from_exception(
        subject=subject,
        subject_argv=subject_argv,
        agent=agent,
        exc_info=exc_info,
        burnin_steps=burnin_steps,
        max_steps=max_steps,
    )

  rprint(rf"[bold green]Success:[/] {subject}")
  rprint(
      "Triangulate did not run because no exception was thrown.",
      style="bold green",
  )
  return NoExceptionRaised(subject=subject, subject_argv=subject_argv)
