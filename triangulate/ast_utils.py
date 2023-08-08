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

"""AST utilities."""

import abc
import ast
from collections.abc import Iterable, Sequence
import dataclasses
from typing import cast, Callable, Type

from triangulate import instrumentation_utils

NodeSelector = Callable[[ast.AST], bool]


class NodePredicate(abc.ABC):
  """An AST node predicate."""

  def __call__(self, node: ast.AST) -> bool:
    raise NotImplementedError()

  def __and__(self, other: "NodePredicate") -> "NodePredicate":
    return All((self, other))


class StaticNodePredicate(NodePredicate, abc.ABC):
  """A static predicate function, with no instance attributes.

  Static predicate functions could be defined as top-level functions. Instead,
  they are designed as `NodePredicate` subclasses with a `matches` classmethod
  for these reasons:

  - Inheriting `NodePredicate` methods like `__and__`: not straightforward with
    a top-level function.
  - Efficient static function calls, to avoid creating instance objects just to
    call `__call__`.
  """

  @classmethod
  @abc.abstractmethod
  def matches(cls, node: ast.AST) -> bool:
    raise NotImplementedError()

  def __call__(self, node: ast.AST) -> bool:
    return self.matches(node)


class All(NodePredicate):
  """Returns True if all given predicates return True."""

  def __init__(self, predicates: Sequence[NodePredicate]):
    self.predicates = predicates

  def __call__(self, node: ast.AST) -> bool:
    return all(predicate(node) for predicate in self.predicates)


@dataclasses.dataclass
class HasType(NodePredicate):
  """Returns True for nodes with a given type."""

  type: Type[ast.AST]

  def __call__(self, node: ast.AST) -> bool:
    return isinstance(node, self.type)


class HasLocationInfo(StaticNodePredicate):
  """Returns True for nodes that have location information."""

  @classmethod
  def matches(cls, node: ast.AST) -> bool:
    return hasattr(node, "lineno") and hasattr(node, "end_lineno")


@dataclasses.dataclass
class OverlapsWithLineNumber(NodePredicate):
  """Returns True for nodes that overlap with a line number."""

  lineno: int

  def __call__(self, node: ast.AST) -> bool:
    if not HasLocationInfo.matches(node):
      return False
    return node.lineno <= self.lineno <= node.end_lineno


class IsProbeStatement(StaticNodePredicate):
  """Returns True for probe statements."""

  @classmethod
  def matches(cls, node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr):
      return False
    if not isinstance(node.value, ast.Call):
      return False
    if (
        not isinstance(node.value.func, ast.Name)
        or node.value.func.id != instrumentation_utils.PROBE_FUNCTION_NAME
    ):
      return False
    return True


class AST:
  """ast.AST wrapper with methods for querying nodes by selector."""

  def __init__(self, source: str):
    self.source = source
    self.root = ast.parse(source)
    self.nodes = [self.root]
    for node in ast.walk(self.root):
      self.nodes.append(node)

  def select_nodes(self, selector: NodeSelector) -> Iterable[ast.AST]:
    return (node for node in self.nodes if selector(node))


def extract_assert_statements(
    source: str, lineno: int | None = None
) -> Sequence[ast.Assert]:
  tree = AST(source)
  selector = HasType(ast.Assert)
  if lineno is not None:
    selector &= OverlapsWithLineNumber(lineno)
  assert_nodes = tuple(tree.select_nodes(selector))
  return cast(tuple[ast.Assert], assert_nodes)


def extract_illegal_state_expression(
    source: str, lineno: int | None = None
) -> str | None:
  # Strategy: extract `assert` test expressions.
  assert_nodes = extract_assert_statements(source, lineno)
  if assert_nodes:
    assert_node = assert_nodes[0]
    assert_condition = ast.unparse(assert_node.test)
    return assert_condition
  # Fallback: return None.
  return None


def is_assert_statement(statement: str) -> bool:
  try:
    parsed = ast.parse(statement.strip(" \n\t"))
    return isinstance(parsed.body[0], ast.Assert)
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def extract_assert_expression(statement: str) -> str:
  parsed = ast.parse(statement.strip(" \n\t"))
  assert_node = parsed.body[0]
  expression = assert_node.test  # pytype: disable=attribute-error
  return ast.unparse(expression)


class LineVisitor(ast.NodeVisitor):
  """Visit intra-statement lines in a Python script.

  Attributes:
    insertion_points:
  """

  def __init__(self):
    self.insertion_points = []

  def visit(self, node: ast.AST) -> None:
    # Skip multiline string literals
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
      return
    # Skip imports
    elif isinstance(node, ast.Import):
      return
    elif hasattr(node, "lineno"):
      # TODO(danielzheng): Log probe candidates.
      # print("Found node with lineno", node.lineno, node.end_lineno, node)
      # print()
      # print("  ", repr(ast.unparse(node)))
      # print("  ", ast.dump(node))
      # print("  ", IsProbeStatement.matches(node))
      # print()
      # Skip probe statements
      if IsProbeStatement.matches(node):
        return
      # TODO(danielzheng): Skip AST nodes that have already been probed.
      self.insertion_points.append(node.lineno - 1)
    if isinstance(
        node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)
    ):
      self.generic_visit(node)


def get_insertion_points(tree: ast.AST) -> list[int]:
  # Collect insertion points
  visitor = LineVisitor()
  visitor.visit(tree)
  insertion_points = visitor.insertion_points
  # TODO(danielzheng): To be robust, insertion point needs to be
  # dominance-aware. Maybe we can use dynamic analysis (e.g. tracing with dummy
  # values) to find valid insertion points along the execution path. Need a
  # Tracer object forwarding all Python operations to underlying value, but
  # keeping track of branch points.
  #
  # For checking probe validity: can keep track of all locals() names at each
  # probe location candidate, to see if all probed names are within scope.
  insertion_points = [x for x in insertion_points if x >= 0]

  if not insertion_points:
    raise ValueError("No valid insertion points found.")

  return insertion_points


class IdentifierExtractor(ast.NodeVisitor):
  """This visitor extracts variables from an AST."""

  def __init__(self):
    self.identifiers = set()

  # This Google-violating function naming is required
  # to confirm with Python's builtin AST library's interface.
  def visit_Name(self, node: ast.Name) -> None:  # pylint: disable=invalid-name
    self.identifiers.add(node.id)

  def visit_Call(self, node: ast.Call) -> None:  # pylint: disable=invalid-name
    for arg in node.args:
      self.visit(arg)


def extract_identifiers(expr: str) -> set[str]:
  """Parse a Python expression and extract its identifiers."""
  root = ast.parse(expr)
  visitor = IdentifierExtractor()
  visitor.visit(root)
  return visitor.identifiers
