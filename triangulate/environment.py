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

"""Bug localization environment."""

import abc
import ast
from collections.abc import Sequence
import dataclasses
import math
import os

from absl import logging
import ordered_set
from python_graphs import program_graph
from python_graphs import program_graph_dataclasses as pb
from rich.syntax import Syntax

from triangulate import ast_utils
from triangulate import exceptions
from triangulate import instrumentation_utils
from triangulate import logging_utils
from triangulate import str_utils

OrderedSet = ordered_set.OrderedSet
ProgramGraph = program_graph.ProgramGraph
ProgramGraphNode = program_graph.ProgramGraphNode

CONSOLE = logging_utils.CONSOLE
rprint = CONSOLE.print
print_horizontal_line = logging_utils.print_horizontal_line
print_panel = logging_utils.print_panel


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
  inspected_variables: OrderedSet[ProgramGraphNode] = dataclasses.field(
      default_factory=OrderedSet
  )
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
        raise exceptions.LineNumberOutOfBoundsError(self.code, self.bug_lineno)

    # TODO(danielzheng): Make this condition precise.
    if self.illegal_state_expression is None:
      illegal_state_expression = ast_utils.extract_illegal_state_expression(
          self.code, self.bug_lineno
      )
      if illegal_state_expression is None:
        raise exceptions.CouldNotIdentifyIllegalStateExpressionError(
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
      raise exceptions.VariableNotInspectionCandidateError(
          variable,
          self.inspected_variables,
          self.candidate_variables_to_inspect,
      )

    if variable in self.inspected_variables:
      raise exceptions.VariableAlreadyInspectedError(
          variable,
          self.inspected_variables,
          self.candidate_variables_to_inspect,
      )

    new_inspected_variables = self.inspected_variables | {variable}
    new_candidate_variables_to_inspect = OrderedSet(
        self.candidate_variables_to_inspect - {variable}
    )

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
            new_candidate_variables_to_inspect.update(
                tuple(return_value_identifiers)
            )

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
      indentation = " " * str_utils.leading_whitespace_count(preceding_line)
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


@dataclasses.dataclass
class InspectVariables(Action):
  """Update state by inspecting variables in the frontier.

  Attributes:
    variables_to_inspect: Sequence of variables (represented as ast.Name program
    graph nodes) to inspect.
  """

  variables_to_inspect: Sequence[ProgramGraphNode]

  def update(self, state: State) -> State:
    """Updates state by inspecting variables."""
    return state.inspect_variables(self.variables_to_inspect)


################################################################################
# Bug localization environment
################################################################################


Reward = int


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
  """

  def __init__(
      self,
      subject: str | os.PathLike[str],
      subject_argv: Sequence[str],
      bug_lineno: int | None = None,
      burnin_steps: int | None = None,
      max_steps: int | None = None,
  ):
    """Construct an environment instance."""
    self.subject = str(subject)
    self.subject_argv = subject_argv
    self.subject_output = set()
    self.steps = 0
    self.max_steps = max_steps if max_steps is not None else 100
    if burnin_steps:
      self.max_burnin_steps = math.ceil(burnin_steps * self.max_steps)
    else:
      self.max_burnin_steps = self.max_steps
    try:
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
