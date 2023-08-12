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

"""String utilities."""


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
