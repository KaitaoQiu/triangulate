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

"""Tests for ast_utils."""

import ast
from collections.abc import Sequence

from absl.testing import absltest
from absl.testing import parameterized

from triangulate import ast_utils


EXAMPLE_PROGRAM = """\
def add_one(x):
    triangulate_probe(['x'], locals())
    assert x == 1
    return x + 1
""".strip()


class ASTTest(absltest.TestCase):

  def test_get_insertion_points(self):
    tree = ast.parse(EXAMPLE_PROGRAM)
    insertion_line_numbers = ast_utils.get_insertion_points(tree)
    # Every line except the debugging probe call is an insertion point.
    self.assertSequenceEqual(insertion_line_numbers, [0, 2, 3])

  def test_extract_identifiers(self):
    test_expr = "x + y * foo(z,c)"
    test_expr_fv = set(["c", "x", "y", "z"])
    fv = ast_utils.extract_identifiers(test_expr)
    self.assertEqual(test_expr_fv, set(fv))


class ASTNodeSelection(parameterized.TestCase):
  """AST node selection tests."""

  def setUp(self):
    super().setUp()
    self.tree = ast.parse(EXAMPLE_PROGRAM)

  @parameterized.named_parameters(
      dict(
          testcase_name="has_type_return",
          node_predicate=ast_utils.HasType(ast.Return),
          expected_node_texts=["return x + 1"],
      ),
      dict(
          testcase_name="overlaps_with_line_number",
          node_predicate=ast_utils.OverlapsWithLineNumber(1),
          expected_node_texts=[EXAMPLE_PROGRAM, "x"],
      ),
      dict(
          testcase_name="is_probe_statement",
          node_predicate=ast_utils.IsProbeStatement(),
          expected_node_texts=["triangulate_probe(['x'], locals())"],
      ),
  )
  def test_select_nodes(
      self,
      node_predicate: ast_utils.NodePredicate,
      expected_node_texts: Sequence[str],
  ):
    nodes = ast_utils.AST(EXAMPLE_PROGRAM).select_nodes(node_predicate)
    node_texts = tuple(ast.unparse(node) for node in nodes)
    self.assertSequenceEqual(node_texts, expected_node_texts)


if __name__ == "__main__":
  absltest.main()
