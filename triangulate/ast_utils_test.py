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

"""Tests for third_party.py.triangulate."""

import ast
from collections.abc import Sequence
import os

from absl.testing import absltest
from absl.testing import parameterized

from triangulate import ast_utils

TESTDATA_DIRECTORY = os.path.join(
    absltest.get_default_test_srcdir(),
    "triangulate/testdata",
)
TEST_PROGRAM_PATH = os.path.join(TESTDATA_DIRECTORY, "quoter.py")


class ASTTest(absltest.TestCase):

  def test_get_insertion_points(self):
    with open(TEST_PROGRAM_PATH, "r", encoding="utf-8") as f:
      source = f.read()
    tree = ast.parse(source)
    insertion_points = ast_utils.get_insertion_points(tree)
    print(insertion_points)  # Debugging, not sure of the current value
    # TODO(etbarr): Verify whether `insertion_points` is correct.
    self.assertLen(insertion_points, 7)

  def test_extract_identifiers(self):
    test_expr = "x + y * foo(z,c)"
    test_expr_fv = set(["c", "x", "y", "z"])
    fv = ast_utils.extract_identifiers(test_expr)
    self.assertEqual(test_expr_fv, set(fv))


EXAMPLE_PROGRAM = """\
def add_one(x):
    triangulate_probe(['x'], locals())
    assert x == 1
    return x + 1
""".strip()


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
