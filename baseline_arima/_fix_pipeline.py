"""Fix ALL non-ASCII characters in pipeline_arima_multilevel.py"""
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "pipeline_arima_multilevel.py")

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Build cleaned content - replace ALL non-ASCII with ASCII equivalents
replacements = {
    "\u2500": "-",    # box-drawing horizontal line
    "\u2014": "--",   # em dash
    "\u2013": "-",    # en dash
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2026": "...",  # ellipsis
    "\u2192": "->",   # arrow
}
for uni, asc in replacements.items():
    content = content.replace(uni, asc)

# Also fix the split print in run_all if it exists
old = ('        print(f"  Test  : d_{self.test_start_d} -> d_{self.test_start_d + self.test_days - 1} "\n'
       '              f"({self.test_days} days)")\n')
new = ('        print(f"  Test  : d_{self.test_start_d} -> d_{self.test_start_d + self.test_days - 1} ({self.test_days} days)")\n')
content = content.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

bad = [(i, c) for i, c in enumerate(content) if ord(c) > 127]
if bad:
    print(f"Still {len(bad)} non-ASCII chars:")
    for i, c in bad[:5]:
        print(f"  pos={i} U+{ord(c):04X} char={repr(c)}")
else:
    print("OK -- no non-ASCII characters remaining.")
