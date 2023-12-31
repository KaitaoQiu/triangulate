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

"""Single function call bug localization example.

Requires visiting multiple variables via the data flow graph.
"""

import argparse
import math


def main(x1: int, y1: int, x2: int, y2: int):
  d = distance(x1=x1, x2=x2, y1=y1, y2=y2)
  assert d > 0, "Distance must be positive."
  assert d > 10, "Distance must be greater than 10."


def distance(x1: int, y1: int, x2: int, y2: int):
  """Returns the Euclidean distance between two points."""
  x_diff = x1 - x2
  y_diff = y1 - y2
  x_diff_squared = x_diff * x_diff
  y_diff_squared = y_diff * y_diff
  return math.sqrt(x_diff_squared + y_diff_squared)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("x1", type=int)
  parser.add_argument("y1", type=int)
  parser.add_argument("x2", type=int)
  parser.add_argument("y2", type=int)

  args = parser.parse_args()
  print(f"Parsed arguments: {args}")
  main(x1=args.x1, y1=args.y1, x2=args.x2, y2=args.y2)
