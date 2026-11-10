# usage: python3 style_header.py <git-ref>   (run from repo root; reads blobs via git, no checkout)
import subprocess, sys, re, collections, json
ref = sys.argv[1]
canon_tail = """
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.""".lstrip("\n").splitlines()
files = [f for f in subprocess.check_output(["git", "ls-tree", "-r", "--name-only", ref, "src", "tests"], text=True).split() if f.endswith(".py")]
res = collections.Counter(); out = []
for f in files:
    txt = subprocess.check_output(["git", "show", f"{ref}:{f}"], text=True)
    lines = txt.splitlines()
    first = lines[0] if lines else ""
    yrs = subprocess.check_output(["git", "log", ref, "--follow", "--diff-filter=A", "--format=%ad", "--date=format:%Y", "--", f], text=True).split()
    add_year = yrs[-1] if yrs else None
    yrs2 = subprocess.check_output(["git", "log", ref, "--no-renames", "--diff-filter=A", "--format=%ad", "--date=format:%Y", "--", f], text=True).split()
    add_year2 = yrs2[-1] if yrs2 else None
    m = re.match(r"# Copyright (\d{4}) (.*)", first)
    if not txt.strip():
        kind = "empty_file"
    elif not m:
        kind = "no_header"
    elif "Rebellions" not in m.group(2):
        kind = "third_party_header"
    else:
        exact = m.group(2) == "Rebellions Inc. All rights reserved." and lines[1:2] == [""] and lines[2:2+len(canon_tail)] == canon_tail
        kind = "rebellions_canonical" if exact else "rebellions_noncanonical"
        if exact and m.group(1) not in (add_year, add_year2):
            kind = "rebellions_canonical_year_mismatch"
    res[kind] += 1
    out.append({"path": f, "kind": kind, "first": first[:120], "header_year": m.group(1) if m else None, "add_year": add_year, "add_year_norename": add_year2})
print(dict(res), "total", len(files))
for o in out:
    if o["kind"] not in ("rebellions_canonical", "third_party_header"):
        print(o["kind"], o["path"], "|", o["first"], "| hdr", o["header_year"], "add", o["add_year"], o["add_year_norename"])
