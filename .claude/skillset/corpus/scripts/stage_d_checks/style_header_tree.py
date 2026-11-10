# usage (from repo root): python3 style_header_tree.py <tree_dir> <git_ref_for_history> [fallback_year]
# Header text is read from <tree_dir>; the add year comes from `git log <ref> --diff-filter=A` (with and without --follow);
# files absent from <ref> history get fallback_year (they were added after <ref>).
import subprocess, sys, re, collections, json, pathlib
tree = pathlib.Path(sys.argv[1]); ref = sys.argv[2]; fallback = sys.argv[3] if len(sys.argv) > 3 else None
canon_tail = ['# Licensed under the Apache License, Version 2.0 (the "License");', '# you may not use this file except in compliance with the License.', '# You may obtain a copy of the License at:', '', '#     http://www.apache.org/licenses/LICENSE-2.0', '', '# Unless required by applicable law or agreed to in writing, software', '# distributed under the License is distributed on an "AS IS" BASIS,', '# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.', '# See the License for the specific language governing permissions and', '# limitations under the License.']
def years(f, follow):
    args = ["git", "log", ref] + (["--follow"] if follow else ["--no-renames"]) + ["--diff-filter=A", "--format=%ad", "--date=format:%Y", "--", f]
    y = subprocess.check_output(args, text=True).split()
    return y[-1] if y else fallback
out = []; c = collections.Counter()
for p in sorted(list((tree / "src").rglob("*.py")) + list((tree / "tests").rglob("*.py"))):
    f = str(p.relative_to(tree)); txt = p.read_text(); lines = txt.splitlines()
    first = lines[0] if lines else ""
    if first.startswith("#!"): lines = lines[1:]; first = lines[0] if lines else ""
    y1, y2 = years(f, True), years(f, False)
    m = re.match(r"# Copyright (\d{4}) (.*)", first)
    if not txt.strip(): kind = "empty_file"
    elif not m: kind = "no_header"
    elif "Rebellions" not in m.group(2): kind = "third_party_header"
    elif not (m.group(2) == "Rebellions Inc. All rights reserved." and lines[1:2] == [""] and lines[2:13] == canon_tail):
        kind = "rebellions_noncanonical_format"
    elif m.group(1) in (y1, y2): kind = "ok"
    elif y1 == "2023" and y2 in ("2023", "2025") and m.group(1) == "2025": kind = "ok_bulk_relicensed_2025"  # initial import, relicensed by #19
    else: kind = "year_mismatch"
    c[(f.split("/")[0], kind)] += 1
    out.append(dict(path=f, kind=kind, first=first[:120], header_year=m.group(1) if m else None, add_follow=y1, add_norename=y2))
for k, v in sorted(c.items()): print(*k, v)
json.dump(out, open(sys.argv[4] if len(sys.argv) > 4 else "/dev/null", "w"), indent=0)
for o in out:
    if o["path"].startswith("src/") and not o["kind"].startswith("ok") and o["kind"] != "third_party_header":
        print(o["kind"], o["path"], "|", o["first"], "| hdr", o["header_year"], "add", o["add_follow"], o["add_norename"])
