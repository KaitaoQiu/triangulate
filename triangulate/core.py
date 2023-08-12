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

"""Bug localization entry point functions."""

from collections.abc import Sequence
import contextlib
import dataclasses
import io
import os
import runpy
import sys
import traceback
import types
from typing import TypeAlias

from triangulate import agents
from triangulate import environment
from triangulate import exceptions
from triangulate import logging_utils

Agent = agents.Agent
Action = environment.Action
Environment = environment.Environment
State = environment.State

CONSOLE = logging_utils.CONSOLE
rprint = logging_utils.rprint
print_panel = logging_utils.print_panel


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
ResultOrError: TypeAlias = Result | exceptions.BugLocalizationError


# TODO(danielzheng): Fix this to take `illegal_state_expression` instead of
# `bug_lineno`. `bug_lineno` is imprecise and insufficient.
def run_with_bug_lineno(
    subject: str | os.PathLike[str],
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
    if isinstance(action, environment.Halt):
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
    subject: str | os.PathLike[str],
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
    Bug localization result: `CouldNotIdentifyIllegalStateExpressionError` if
    the illegal state expression could not be identified from the exception, or
    the result of `run_with_bug_lineno` otherwise.
  """
  rprint("Exception caught:", style="bold yellow")
  _, exc_value, tb = exc_info
  if exc_value is None:
    raise exceptions.SubjectProgramNoExceptionRaisedError(
        subject=subject, subject_argv=subject_argv
    )
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
  except exceptions.CouldNotIdentifyIllegalStateExpressionError as e:
    rprint(
        "Could not identify illegal state expression from exception:",
        style="bold red",
    )
    traceback.print_exception(exc_value, limit=-1)
    return e


def run(
    subject: str | os.PathLike[str],
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
      runpy.run_path(str(subject), run_name="__main__")
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
  return exceptions.SubjectProgramNoExceptionRaisedError(
      subject=subject, subject_argv=subject_argv
  )
