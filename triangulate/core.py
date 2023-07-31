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

"""This is executable pseudocode for an RL localiser."""

import ast
import contextlib
import io
import math
import os
import tempfile
from typing import List, TextIO, Tuple

from absl import logging
import numpy as np
from triangulate import ast_utils
from triangulate import sampling_utils

rng = np.random.default_rng(seed=654)

################################################################################
# Utils
################################################################################


# TODO(etbarr):  Rewrite to use AST.
def write_lines_to_file(
    target_file: TextIO, offset_lines: List[Tuple[int, str]]
) -> None:
  """Write lines to their paired offset in the target file.

  Args:
      target_file:  File descriptor to which the to write the lines to their
        paired offsets
      offset_lines:  A list of lines paired with a target offset
  """
  for offset, line in offset_lines:
    target_file.seek(offset)
    target_file.write(line)


################################################################################
# Barebones RL
################################################################################


class State:
  """State generated by the environment and passed to the agent in RL loop.

  Attributes: 
    codeview : [str] code lines in current agent window 
    focal_expr : str current expression
    subject_with_probes: file descriptor of program being debugged
    probes: [str x int] list of probes, which pair a query and an offset.
  """

  def set_ise(self, ise: str) -> None:
    try:
      compile(ise, "<string>", "eval")
    except SyntaxError as e:
      err_template = "Error: %s is an invalid Python expression."
      logging.error(err_template, ise)
      # TODO(etbarr) add when Python 3.11 is available within Google
      # e.add_note(err_template % expr)
      raise e
    self.ise = ise

  def set_focal_expr(self, focal_expr: str) -> None:
    try:
      compile(focal_expr, "<string>", "eval")
    except SyntaxError as e:
      err_template = "Error: %s is an invalid Python expression."
      logging.error(err_template, focal_expr)
      # TODO(etbarr) add when Python 3.11 is available within Google
      # e.add_note(err_template % expr)
      raise e
    self.focal_expr = focal_expr

  def __init__(
      self,
      subject_with_probes: TextIO,
      bug_trap: int,
      probes: List[Tuple[int, str]] | None = None,
  ):
    self.codeview = subject_with_probes.readlines()  # TODO(etbarr): exceptions?
    error_message = "bug trap out of bounds"
    assert 0 <= bug_trap < len(self.codeview), error_message
    if ast_utils.is_assert_statement(self.codeview[bug_trap]):
      self.set_ise(ast_utils.extract_assert_expression(self.codeview[bug_trap]))
    else:
      raise ValueError(
          "Bug_trap must identify an assertion statement, but"
          f" codeview[bug_trap={bug_trap}] ="
          f" '{self.codeview[bug_trap].strip()}', which is not."
      )
    focal_expr = ast_utils.extract_assert_expression(self.codeview[bug_trap])
    self.set_focal_expr(focal_expr)
    self.subject_with_probes = subject_with_probes
    if probes is None:
      self.probes = []
    else:
      self.probes = probes

  def get_illegal_state_expr_ids(self):
    """Return identifiers in the illegal state expression.

    Returns:
        Identifiers in the illegal state expression
    """
    return ast_utils.extract_identifiers(self.ise)

  def illegal_bindings(self) -> str | None:
    """Return f-string for reporting illegal bindings.

    Returns:
        Returns an f-string over the illegal bindings
    """
    idents = self.get_illegal_state_expr_ids()
    if not idents:
      return None
    ident = idents.pop()
    bindings = f"{ident} = " + "{" + f"{ident}" + "}"
    for ident in idents:
      bindings += f", {ident} = " + "{" + f"{ident}" + "}"
    return bindings

  def get_codeview(self) -> List[str]:
    """Return codeview."""
    return self.codeview

  def to_string(self):
    """Convert object into string representation.

    Returns:
        Object contents serialised into a string.
    """
    print(self.subject_with_probes, self.codeview)
    # TODO(etbarr): implement.


class Agent:
  """Baseclass for a minimal RL agent.

  Attributes: total_reward : int the reward accumulator env : the agent's
  environment
  """

  def __init__(self, env, total_reward: int = 0):
    """Agent constructor.

    Args:
        env: handle to the environment.
        total_reward: accumulated reward

    Returns:
        An agent instance
    """
    self.env = env
    self.total_reward = total_reward

  def pick_action(self, state: State, reward: int) -> None:
    """Pick an action given the current state and reward.

    Args:
        state: Current state
        reward:  Reward for last action and current state

    Returns:
        None
    """
    print(
        f"abstract method, not sure it's needed; {state} {reward}",
        state,
        reward,
    )

  def add_probes(self, state: State, probes: List[Tuple[int, str]]) -> None:
    """Add probes to the codeview of the state.

    Args:
        state: Current state
        probes:  list of probes, which pair queries and offsets

    Returns:
        None
    """
    for offset, probe in probes:
      state.codeview.insert(offset, probe)
    state.subject_with_probes.seek(0)
    state.codeview.seek(0)
    state.subject_with_probes.writelines(state.codeview)

  def repr(self) -> str:
    """Convert object into string representation.

    Returns:
        Object contents serialised into a string.
    """
    return str(self.total_reward)


class Localiser(Agent):
  """Represent the localiser agent.

  Attributes:
    codeview : [str] code lines in current agent window 
    ise : str illegal state expression 
    focal_expression : str current expression
    subject_with_probes: file descriptor to copy of program being debugged
    probes: [str x int] list of probes, which pair a query and an offset.

  Methods:
      generate_probes(self, state) -> []:
      pick_action(self, state : State, reward: int) -> None:
  """

  def _generate_probes_random(self, state):
    """Generate probes for the given state.

    Args:
      state: current state

    Returns:
      List of probes, which pair queries and offsets
    """

    state.subject_with_probes.seek(0)
    tree = ast.parse(state.subject_with_probes.read())
    insertion_points = ast_utils.get_insertion_points(tree)
    samples = sampling_utils.sample_zipfian(1, len(insertion_points))
    offsets = sampling_utils.sample_wo_replacement_uniform(
        samples[0], insertion_points
    )
    offsets.sort()

    ise = (
        f"Illegal state predicate: '{state.ise}' = "
        + "{eval("
        + repr(state.ise)
        + ")}; "
    )
    isb = f"bindings: {state.illegal_bindings()}"
    query = 'f"' + ise + isb + '"'

    probes = []
    for offset in offsets:
      probes.append((offset, f"print({query})\n"))
    state.probes = probes

    return probes

  # TODO(etbarr): Build AST, reverse its edges and walk the tree from focal
  #       expression to control expressions and defs
  # Ignore aliases for now.
  def _generate_probes_baseline(self, state):
    """Use analysis techniques to generate probes for the given state.

    Args:
      state:

    Raises:
      NotImplementedError
    """
    raise NotImplementedError(f"Not implementated; {state}", state)

  # Answers two questions:  decides 1) where to query 2) what.
  # Returns list of probes
  def generate_probes(self, state):
    """Generate probes for the given state.

    To create each probe this, function must decide whether to query what.

    Args:
        state: current state

    Returns:
        Object contents serialised into a string.
    """
    return self._generate_probes_random(state)

  def pick_action(self, state, reward: int) -> None:
    """Pick action in state.

    Args:
        state: current state
        reward:  the reward for the previous state
    """
    # TODO(etbarr):  add action selection
    # pp.pprint(f"state.codeview = {state.codeview}, reward = {reward},
    #          self.total_reward = {self.total_reward}")
    self.add_probes(state, self.generate_probes(state))
    self.total_reward += reward
    self.env.live = False


class Environment:
  """Represent the RL environment.

  Attributes:
      subject: str
      subject_output: str
      steps: int
      max_burnin: int
      max_steps: int
      subject_with_probes: TextIO
      state: State
  """

  def __init__(
      self,
      subject: str,
      bug_triggering_input: str,
      bug_trap: int,
      burnin: int,
      max_steps: int,
      probe_output_filename: str,
      loglevel: int,
  ):
    """Construct an environment instance.

    Although we instrument the buggy program with probes, these probes write
    their output to a new temporary file, leaving the buggy program's output
    unchanged.  Thus, if we do detect a change in that output, there is an
    error in the instrumentation.
    TODO(etbarr):  Add the argument, with a default, for the probe output
    file.

    Args:
        subject: str,
        bug_triggering_input: str,
        bug_trap: int,
        burnin: int,
        max_steps: int,
        probe_output_filename: str,
        loglevel: int = 0,

    Returns:
        An environment instance
    """
    del bug_triggering_input, probe_output_filename  # Unused for now.
    self.subject = subject
    self.subject_output = set()
    self.subject_with_probes = None
    self.steps = 0
    self.max_steps = max_steps
    if burnin != 0:
      self.max_burnin = math.ceil(burnin * self.max_steps)
    else:
      self.max_burnin = max_steps
    file_extension = os.path.splitext(self.subject)[1]
    # TODO(etbarr) bl/284330538 fix extension kludge
    if file_extension != ".py":
      err_template = "Error: %s is not a Python script."
      logging.error(err_template, self.subject)
      raise ValueError(err_template, self.subject)
    try:
      # pylint: disable=consider-using-with
      if loglevel == logging.DEBUG:
        self.subject_with_probes = tempfile.NamedTemporaryFile(
                mode="r+", delete=True
        )
        print(
          f"The subject with probes saved to {subject_with_probes.name}."
        )
      else:
        self.subject_with_probes = tempfile.NamedTemporaryFile(
                mode="r+", delete=False
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
    self.state = State(self.subject_with_probes, bug_trap)

    self.subject_output.add(self.execute_subject())

  # TODO(etbarr) Gather and pass a subject's parameters to it.
  def execute_subject(self) -> str:
    """Execute an instrumented version of the buggy program.

    Returns:
      Returns the subject's output, concatenating standard and error.

    Raises:
      CallProcessError if subprocess.run fails.
    """
    assert self.subject_with_probes is not None
    self.subject_with_probes.seek(0)

    python_source = self.subject_with_probes.read()
    try:
      compiled_source = compile(
          python_source, "<code_to_instrument>", mode="exec"
      )
    except SyntaxError as e:
      raise e

    try:
      exec_globals = {}
      exec_locals = None
      buffer = io.StringIO()
      with (
          contextlib.redirect_stdout(buffer),
          contextlib.redirect_stderr(buffer),
      ):
        exec(compiled_source, exec_globals, exec_locals)  # pylint:disable=exec-used
      return buffer.getvalue()
    except Exception as e:
      logging.error("Error: %s", e)
      raise e

  def reward(self) -> int:
    """Return reward for current state.

    Returns:
        reward
    """
    # TODO(etbarr)
    return 1

  def terminate(self) -> bool:
    """Determine whether to terminate simulation.

    Returns:
        termination condition
    """

    if self.steps >= self.max_steps:
      return True
    return False

  def update(self, action) -> None:
    """Update simulation given the selected action.

    Args:
        action:  action selected by agent

    Raises:
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
    if self.steps > self.max_burnin:
      error_message = (
          "Error: probe insertion or execution changed program semantics."
      )
      if stdouterr not in self.subject_output:
        logging.exception(error_message)
        raise AssertionError(error_message)

    self.subject_output.add(stdouterr)
    # TODO(etbarr) Create and return a new state instance
    # Probe's write their output to a fresh file

  def to_string(self) -> str:
    """Convert object into string representation.

    Returns:
        Object contents serialised into a string.
    """
    raise NotImplementedError
