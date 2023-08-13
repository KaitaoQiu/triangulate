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

"""Bug localization agents."""

import abc
import ast
from collections.abc import Sequence, Set
import dataclasses
import enum
import queue
from typing import Callable, Generic, TypeVar

import ordered_set
from python_graphs import program_graph

from triangulate import ast_utils
from triangulate import environment
from triangulate import instrumentation_utils
from triangulate import logging_utils
from triangulate import sampling_utils

OrderedSet = ordered_set.OrderedSet
ProgramGraph = program_graph.ProgramGraph
ProgramGraphNode = program_graph.ProgramGraphNode

Action = environment.Action
Environment = environment.Environment
Probe = environment.Probe
Probes = environment.Probes
Reward = environment.Reward
State = environment.State

# Actions
AddProbes = environment.AddProbes
Halt = environment.Halt
InspectVariables = environment.InspectVariables

rprint = logging_utils.rprint

################################################################################
# Bug localization agents
################################################################################


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
        n.node.id for n in state.illegal_state_expression_identifiers()
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
    return InspectVariables(variables)

  @property
  def visited_variables(self) -> Set[str]:
    return self.strategy.visited


def make_search_agent_factory(
    strategy_type: type[SearchStrategy],
) -> Callable[[], SearchAgent]:
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
