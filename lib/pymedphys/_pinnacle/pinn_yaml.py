# Copyright (C) 2019 South Western Sydney Local Health District,
# University of New South Wales

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This work is derived from:
# https://github.com/AndrewWAlexander/Pinnacle-tar-DICOM
# which is released under the following license:

# Copyright (c) [2017] [Colleen Henschel, Andrew Alexander]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import logging
import re

from pymedphys._imports import yaml

logger = logging.getLogger(__name__)


def _sanitise_line(line):
    """Clean a single line to avoid YAML parse failures.

    Handles several edge cases seen in older Pinnacle archives:
    - Unquoted values containing special YAML characters (: # [ ] { })
    - Backslash sequences that confuse the YAML parser
    - Trailing whitespace / carriage returns
    """
    # Strip trailing whitespace / CR
    line = line.rstrip()

    # If the line contains a key = value or key : value assignment,
    # ensure the value portion is safe for YAML.
    m = re.match(r"^(\s*\S+\s*:\s*)(.*)", line)
    if m:
        prefix, value = m.group(1), m.group(2)
        value = value.rstrip(";").strip()
        if value:
            # Quote the value if it contains YAML-hostile characters and
            # isn't already quoted.
            needs_quoting = (
                not (value.startswith('"') and value.endswith('"'))
                and not (value.startswith("'") and value.endswith("'"))
                and re.search(r"[:\\#\[\]{}]", value)
            )
            if needs_quoting:
                # Escape existing double-quotes inside the value
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                line = f'{prefix}"{escaped}"'
            else:
                line = f"{prefix}{value}"
        else:
            line = prefix

    # Re-add the newline that callers expect
    return line + "\n"


def _fallback_parse(data_lines):
    """Best-effort key-value parser for when YAML conversion fails.

    Reads Pinnacle's pseudo-YAML as flat key = value pairs and nested
    blocks.  This is intentionally simple — it doesn't handle every
    Pinnacle construct, but it's good enough to extract the fields that
    the export pipeline actually reads (patient setup, machine info,
    points, etc.) without crashing.
    """
    result = {}
    stack = [result]  # stack of dicts/lists being built
    list_depths = set()

    for raw_line in data_lines:
        line = raw_line.strip()

        # Skip empty, comment open/close, and closing braces that
        # just terminate blocks.
        if not line or line.startswith("/*") or line.startswith("*/"):
            continue
        if line == "};" or line == "}":
            if len(stack) > 1:
                stack.pop()
            continue

        # Block opener: "Key ={" or "Key = {"
        m = re.match(r"^(\S+)\s*=\s*\{", line)
        if m:
            key = m.group(1)
            indent = len(raw_line) - len(raw_line.lstrip())
            if "Array" in key or "List" in key:
                list_depths.add(indent)
                new_list = []
                if isinstance(stack[-1], dict):
                    stack[-1][key] = new_list
                stack.append(new_list)
            else:
                new_dict = {}
                if isinstance(stack[-1], dict):
                    stack[-1][key] = new_dict
                elif isinstance(stack[-1], list):
                    stack[-1].append(new_dict)
                stack.append(new_dict)
            continue

        # Simple assignment: "Key = Value;"
        m = re.match(r"^(\S+)\s*=\s*(.*?)\s*;?\s*$", line)
        if m:
            key, val = m.group(1), m.group(2)
            # Strip surrounding quotes
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            else:
                # Try numeric conversion
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            if isinstance(stack[-1], dict):
                stack[-1][key] = val
            elif isinstance(stack[-1], list):
                stack[-1].append({key: val})
            continue

    return result


# Pinnacle anonymous numbered object header, e.g. "#0 ={" at column 0.
# Anchored to the line start so that indented (nested) numbered entries are
# left to the existing list/sequence handling in convert_to_yaml.
_NUMBERED_OBJECT_HEADER = re.compile(r"^#\d+\s*={")


class _DuplicateKeyLoader(yaml.SafeLoader):
    """SafeLoader that records duplicate mapping keys instead of hiding them.

    PyYAML silently keeps the last value when a mapping has duplicate keys.
    For Pinnacle files that behaviour turns a structural parse problem into
    silent data corruption — two objects merge and one wins — which is
    exactly how a two-machine Machines file came to look like a single
    machine.  Collecting the duplicates lets the caller warn about it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.duplicate_keys = []

    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                self.duplicate_keys.append(key)
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _safe_load_reporting_duplicates(yaml_text, filename, segment):
    """yaml.safe_load equivalent that warns about duplicate mapping keys."""
    loader = _DuplicateKeyLoader(yaml_text)
    try:
        data = loader.get_single_data()
        duplicates = loader.duplicate_keys
    finally:
        loader.dispose()

    if duplicates:
        unique = sorted(set(duplicates))
        logger.warning(
            "'%s' (segment %d): %d duplicate key(s) were overwritten while "
            "parsing — data from earlier objects has been LOST. Keys: %s%s",
            filename,
            segment,
            len(duplicates),
            ", ".join(repr(k) for k in unique[:10]),
            " …" if len(unique) > 10 else "",
        )
    return data


# if multiple trials, result = list of dicts,  otherwise single dict
def pinn_to_dict(filename):
    result = None
    with open(filename, encoding="ISO-8859-1", errors="ignore") as fp:
        data = fp.readlines()

        if not data:
            return result

        # Split data into smaller chunks, if first line appears more than one
        # Useful for plan.Trial files with more than one Trial
        #
        # Pinnacle also stores anonymous numbered objects with "#N ={"
        # headers ("#0 ={", "#1 ={", ...) — plan.Pinnacle.Machines uses this
        # for its machine list.  Those headers are NOT identical to each
        # other, so the equality test below never split them; and because
        # "#" opens a comment in YAML the headers were then silently dropped,
        # collapsing every object in the file into a single mapping in which
        # duplicate keys overwrote one another (last one wins).  A two-machine
        # Machines file therefore parsed as one machine wearing the second
        # machine's name and a mixture of both machines' data.
        #
        # Splitting on the "#N ={" pattern restores one object per segment.
        # Only files that *begin* with such a header take this path, so
        # nested "#N ={" entries inside "...List ={" containers (which the
        # sequence handling in convert_to_yaml already deals with) are
        # completely unaffected.
        first_line = data[0]
        if _NUMBERED_OBJECT_HEADER.match(first_line):
            indices = [
                i for i, line in enumerate(data) if _NUMBERED_OBJECT_HEADER.match(line)
            ]
            logger.debug(
                "'%s' uses numbered object headers; split into %d object(s).",
                filename,
                len(indices),
            )
        else:
            indices = [i for i, line in enumerate(data) if line == first_line]

        for i, _ in enumerate(indices):
            next_index = -1

            if i + 1 < len(indices):
                next_index = indices[i + 1]

                # If there are multiple segments, return list, otherwise just the dict
                if not isinstance(result, list):
                    result = []

            split_data = data[indices[i] : next_index]

            try:
                yaml_text = convert_to_yaml(split_data)
                d = _safe_load_reporting_duplicates(yaml_text, filename, i)
            except Exception as exc:
                logger.warning(
                    "YAML parse failed for '%s' (segment %d): %s — "
                    "attempting sanitised re-parse",
                    filename,
                    i,
                    exc,
                )
                try:
                    sanitised = [_sanitise_line(l) for l in split_data]
                    yaml_text = convert_to_yaml(sanitised)
                    d = _safe_load_reporting_duplicates(yaml_text, filename, i)
                except Exception as exc2:
                    logger.warning(
                        "Sanitised YAML parse also failed for '%s' (segment %d): %s — "
                        "falling back to line-by-line parser",
                        filename,
                        i,
                        exc2,
                    )
                    d = _fallback_parse(split_data)

            if d is None:
                continue

            if isinstance(result, list):
                if isinstance(d, dict) and len(d) == 1:
                    result.append(d[list(d.keys())[0]])
                else:
                    result.append(d)
            else:
                result = d

    return result


def convert_to_yaml(data):
    out = ""
    listIndents = []
    in_comment = False
    c = 0
    for _, line in enumerate(data, 0):
        # Remove comment blocks.  A comment may open and close on the
        # same line (e.g. ``/* ... */``), so check the close *after*
        # detecting the open rather than on the next iteration.
        if in_comment:
            if re.search(r"\*\/", line):
                in_comment = False
            continue

        if re.search(r"^\/\*", line):
            # Check if the comment also closes on this same line
            if re.search(r"\*\/", line):
                in_comment = False
            else:
                in_comment = True
            continue

        # Get the indentation of this line
        thisIndent = len(re.match(r"^\s*", line).group())

        # Check for start list/array
        if re.search("Array ={", line) or re.search("List ={", line):
            listIndents.append(thisIndent)

        # Check for end list/array
        if re.search("}", line) and thisIndent in listIndents:
            listIndents.pop()

        # If this is the end of an object, discard as not needed for YAML
        if re.search("}", line):
            continue

        # If this line is one indentation in from a start of list,
        # add '-' for YAML sequence
        if thisIndent - 2 in listIndents:
            spaces = " " * thisIndent
            subbed_line = re.sub(r"^\s*", "", line)
            line = f"{spaces}- {subbed_line}"

        # Replace ={ and = with : for assignment
        line = re.sub(" ={", " :", line)
        line = re.sub(" = ", " : ", line)

        # Remove semicolons at end of lines (tolerant of trailing whitespace)
        line = re.sub(r";\s*$", "\n", line)

        out += "" + line
        c += 1

    return out


def pinn_to_yaml(filename):
    with open(filename, encoding="ISO-8859-1", errors="ignore") as fp:
        data = fp.readlines()
        return convert_to_yaml(data)
