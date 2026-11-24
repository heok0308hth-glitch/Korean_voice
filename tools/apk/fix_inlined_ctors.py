"""dex2jar mistranslates R8's inlined constructors (`new-instance C` + `invoke-direct Super.<init>`) as `new Super`.
For a desktop test copy only: give C a constructor with the same signature that delegates to its superclass,
and call that instead. usage: fix_inlined_ctors.py smali_dir"""
import re, glob, os, sys
root = sys.argv[1]
def path(c): return os.path.join(root, c[1:-1] + ".smali")
def slots(sig):
    n = 0; i = 0
    while i < len(sig):
        ch = sig[i]
        if ch in "JD": n += 2; i += 1
        elif ch == "L": n += 1; i = sig.index(";", i) + 1
        elif ch == "[":
            while sig[i] == "[": i += 1
            if sig[i] == "L": i = sig.index(";", i) + 1
            else: i += 1
            n += 1
        else: n += 1; i += 1
    return n
cache = {}
def src(c):
    if c not in cache:
        p = path(c); cache[c] = open(p, encoding="utf-8").read() if os.path.exists(p) else None
    return cache[c]
def ensure_ctor(c, sig, target):
    """Make class c have <init>(sig) chaining to `target`'s <init>(sig)."""
    s = src(c)
    if s is None or c == target: return
    if re.search(r"\.method [^\n]*constructor <init>\(" + re.escape(sig) + r"\)V", s): return
    sup = re.search(r"\.super (L[^;]+;)", s).group(1)
    if sup != target: ensure_ctor(sup, sig, target)
    n = slots(sig) + 1
    call = f"invoke-direct/range {{p0 .. p{n - 1}}}, {sup}-><init>({sig})V" if n > 1 else f"invoke-direct {{p0}}, {sup}-><init>()V"
    cache[c] = s + f"\n.method public constructor <init>({sig})V\n    .registers {n}\n    {call}\n    return-void\n.end method\n"
fixed = 0
for f in glob.glob(root + "/**/*.smali", recursive=True):
    lines = open(f, encoding="utf-8").read().split("\n"); changed = False
    for i, l in enumerate(lines):
        m = re.match(r"\s*new-instance (\w+), (L[^;]+;)", l)
        if not m: continue
        reg, cls = m.group(1), m.group(2)
        for j in range(i + 1, min(i + 40, len(lines))):
            m2 = re.match(r"(\s*invoke-direct(?:/range)? \{" + reg + r"(?:[ ,.][^}]*)?\}, )(L[^;]+;)-><init>\(([^)]*)\)V", lines[j])
            if m2:
                if m2.group(2) != cls:
                    ensure_ctor(cls, m2.group(3), m2.group(2))
                    lines[j] = lines[j].replace(m2.group(2) + "-><init>(", cls + "-><init>(", 1); changed = True; fixed += 1
                break
            if re.match(r"\s*new-instance " + reg + ",", lines[j]): break
    if changed:
        cache.pop(None, None)
        key = "L" + os.path.relpath(f, root)[:-6] + ";"
        cache[key] = "\n".join(lines) if key not in cache else cache[key]
        if key in cache and cache[key] != "\n".join(lines):
            # file already loaded (ctor added earlier): re-apply the call-site edits on top
            base = cache[key]; cache[key] = "\n".join(lines) + base[len(open(f, encoding='utf-8').read()):]
        open(f, "w", encoding="utf-8").write(cache[key])
for c, s in cache.items():
    if s is not None: open(path(c), "w", encoding="utf-8").write(s)
print("call sites fixed", fixed)
