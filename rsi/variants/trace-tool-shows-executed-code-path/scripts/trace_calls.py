# Runs a Python snippet under sys.setprofile and writes a call tree of the functions it executed that live in
# files under ROOT (site-packages, the snippet itself and import-time module code left out).
# Kept compatible with old Python 3 versions (no f-strings).
# usage: trace_calls.py ROOT SNIPPET REPORT MAX_LINES TIME_LIMIT [FOCUS_REGEX]
import os
import re
import signal
import sys
import time
import traceback

# Call order of children matters; plain dicts keep insertion order only from 3.6 on.
if sys.version_info >= (3, 6):
    Children = dict
else:
    from collections import OrderedDict as Children

MAX_NODES = 200000
MAX_RENDER = 20000
MAX_RETS = 3
RET_CHARS = 90


class _Timeout(BaseException):
    pass


class Node(object):
    __slots__ = ("key", "count", "children", "rets", "nret")

    def __init__(self, key):
        self.key = key
        self.count = 0
        self.children = Children()
        self.rets = []
        self.nret = 0


SKIP = Node(None)
OVERFLOW = Node(None)


def safe_repr(v, depth=0):
    # Never runs code from the traced program: only builtin reprs on exact builtin types.
    t = type(v)
    try:
        if v is None:
            s = "None"
        elif t is bool:
            s = "True" if v else "False"
        elif issubclass(t, str):
            s = str.__repr__(v)
            if t is not str:
                s = t.__name__ + "(" + s + ")"
        elif issubclass(t, bytes):
            s = bytes.__repr__(v)
        elif t is int or t is float:
            s = t.__repr__(v)
        elif (t is tuple or t is list) and depth < 2:
            items = [safe_repr(x, depth + 1) for x in (tuple.__iter__(v) if t is tuple else list.__iter__(v))]
            inner = ", ".join(items[:8]) + (", ..." if len(items) > 8 else "")
            s = ("(" + inner + ("," if len(items) == 1 else "") + ")") if t is tuple else ("[" + inner + "]")
        elif t is dict and depth < 2:
            pairs = []
            for k in list(dict.keys(v))[:6]:
                pairs.append(safe_repr(k, depth + 1) + ": " + safe_repr(dict.__getitem__(v, k), depth + 1))
            s = "{" + ", ".join(pairs) + (", ..." if dict.__len__(v) > 6 else "") + "}"
        else:
            s = "<" + t.__name__ + ">"
    except Exception:
        s = "<" + getattr(t, "__name__", "?") + ">"
    if len(s) > RET_CHARS:
        s = s[:RET_CHARS] + "..."
    return s


def main():
    root, snippet, report, max_lines, time_limit = sys.argv[1:6]
    max_lines = max(20, int(max_lines))
    time_limit = float(time_limit)
    focus = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else None
    focus_re = None
    if focus:
        try:
            focus_re = re.compile(focus)
        except re.error:
            focus_re = re.compile(re.escape(focus))

    root_prefix = os.path.realpath(root).rstrip(os.sep) + os.sep
    own = os.path.realpath(__file__)
    snippet_real = os.path.realpath(snippet)
    path_cache = {}
    code_cache = {}
    top = Node(("", 0, "<snippet>"))
    stack = [top]
    frames = [None]
    state = {"nodes": 0, "calls": 0, "modules": 0, "overflow": False}

    def classify(code, frame):
        fn = code.co_filename
        rel = path_cache.get(fn)
        if rel is None:
            rel = False
            if fn and not fn.startswith("<"):
                p = os.path.realpath(fn)
                if (p.startswith(root_prefix) and p != own and p != snippet_real
                        and "/site-packages/" not in p and "/dist-packages/" not in p):
                    rel = p[len(root_prefix):]
            path_cache[fn] = rel
        if rel is False:
            return False
        name = code.co_name
        if name == "<module>":
            return "module"
        qual = getattr(code, "co_qualname", None)
        if qual is None:
            qual = name
            if code.co_argcount > 0 and code.co_varnames[0] in ("self", "cls"):
                try:
                    obj = frame.f_locals.get(code.co_varnames[0])
                    cls = obj if (code.co_varnames[0] == "cls" and type(obj) is type) else type(obj)
                    owner = cls
                    for k in type.mro(cls) if isinstance(cls, type) else ():
                        d = k.__dict__.get(name)
                        f = getattr(d, "__func__", d)
                        if getattr(f, "__code__", None) is code:
                            owner = k
                            break
                    qual = owner.__name__ + "." + name
                except Exception:
                    qual = name
        return (rel, code.co_firstlineno, qual)

    def prof(frame, event, arg):
        if event == "call":
            code = frame.f_code
            info = code_cache.get(code)
            if info is None:
                info = classify(code, frame)
                code_cache[code] = info
            if info is False:
                return
            parent = stack[-1]
            if info == "module":
                state["modules"] += 1
                node = SKIP
            elif parent is SKIP or parent is OVERFLOW:
                node = parent
            else:
                state["calls"] += 1
                node = parent.children.get(info)
                if node is None:
                    if state["nodes"] >= MAX_NODES:
                        state["overflow"] = True
                        node = OVERFLOW
                    else:
                        node = Node(info)
                        parent.children[info] = node
                        state["nodes"] += 1
                node.count += 1
            stack.append(node)
            frames.append(frame)
        elif event == "return":
            if frames[-1] is frame:
                frames.pop()
                node = stack.pop()
                if arg is not None and node is not SKIP and node is not OVERFLOW and node.nret < 50:
                    node.nret += 1
                    if len(node.rets) < MAX_RETS:
                        r = safe_repr(arg)
                        if r not in node.rets:
                            node.rets.append(r)
            elif frame in frames:
                while len(frames) > 1 and frames[-1] is not frame:
                    frames.pop()
                    stack.pop()
                if len(frames) > 1:
                    frames.pop()
                    stack.pop()

    with open(snippet) as f:
        src = f.read()
    try:
        code = compile(src, snippet, "exec")
    except SyntaxError:
        e = sys.exc_info()[1]
        sys.stderr.write("".join(traceback.format_exception_only(type(e), e)))
        with open(report, "w") as f:
            f.write("The snippet did not compile, so nothing ran.\n")
        sys.stderr.flush()
        os._exit(1)
    sys.argv = [snippet]
    sys.path.insert(0, os.getcwd())
    # A fresh __main__ module, so pickling, unittest.main() and similar lookups see the snippet's own names.
    import types
    main_mod = types.ModuleType("__main__")
    main_mod.__file__ = snippet
    main_mod.__builtins__ = __builtins__
    sys.modules["__main__"] = main_mod
    glb = main_mod.__dict__

    def on_alarm(signum, frm):
        raise _Timeout()

    exit_code = 0
    stopped = None
    start = time.time()
    try:
        signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, time_limit)
    except Exception:
        pass
    sys.setprofile(prof)
    try:
        exec(code, glb)
    except _Timeout:
        stopped = "stopped after %.0f s (time limit); the calls below are those made until then" % time_limit
        exit_code = 124
    except SystemExit as e:
        c = e.code
        exit_code = c if isinstance(c, int) else (0 if c is None else 1)
        if not isinstance(c, int) and c is not None:
            sys.stderr.write(str(c) + "\n")
    except BaseException:
        sys.setprofile(None)
        etype, value, tb = sys.exc_info()
        # Drop this script's own frame so the traceback starts in the snippet.
        traceback.print_exception(etype, value, tb.tb_next if tb is not None else tb)
        exit_code = 1
    finally:
        sys.setprofile(None)
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception:
            pass
    elapsed = time.time() - start
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    # Render the aggregated tree. Each line: function, file:line, call count, distinct return values seen.
    rows = []

    def line_for(node, depth):
        rel, lineno, qual = node.key
        s = "  " * depth + qual + "  " + rel + ":" + str(lineno)
        if node.count > 1:
            s += "  x" + str(node.count)
        if node.rets:
            s += "  -> " + " | ".join(node.rets)
            if node.nret > len(node.rets) and len(node.rets) >= MAX_RETS:
                s += " | ..."
        return s

    def matches(node):
        rel, lineno, qual = node.key
        return bool(focus_re.search(qual + " " + rel + ":" + str(lineno)))

    keep = {}

    def mark_focus(node, level):
        # level: None outside any match, else depth below the closest matching ancestor.
        found = False
        for child in node.children.values():
            m = matches(child)
            child_level = 0 if m else (level + 1 if level is not None else None)
            sub = mark_focus(child, child_level)
            if m or sub or (child_level is not None and child_level <= 2):
                keep[id(child)] = True
            if m or sub:
                found = True
        return found

    def walk(node, depth):
        for child in node.children.values():
            if len(rows) >= MAX_RENDER:
                return
            if focus_re is not None and id(child) not in keep:
                continue
            rows.append((depth, child))
            walk(child, depth + 1)

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    if focus_re is not None:
        mark_focus(top, None)
    walk(top, 0)

    out = []
    out.append("exit code %d, %.1f s%s" % (exit_code, elapsed, ("; " + stopped) if stopped else ""))
    head = "repository calls: %d (%d distinct call sites in the tree)" % (state["calls"], state["nodes"])
    if state["modules"]:
        head += "; import-time code of %d module executions not listed" % state["modules"]
    out.append(head)
    if state["overflow"]:
        out.append("(tree too large: calls after %d distinct call sites are counted but not listed)" % MAX_NODES)
    if focus_re is not None:
        out.append("focus %r: matching calls, the calls that led to them and up to 2 levels below them" % focus)
    if not rows:
        if state["calls"] == 0:
            out.append("No function in a file under %s was called (outside import-time code). If the package is "
                       "imported from site-packages instead of this checkout, check sys.path / the install." % root)
        elif focus_re is not None:
            out.append("No recorded call matched the focus pattern; run again without focus to see what ran.")
    else:
        shown = rows
        hidden_note = None
        if len(rows) > max_lines:
            # Keep the shallowest levels that fit, and say how many calls are nested below each cut line.
            depth_counts = {}
            for d, _ in rows:
                depth_counts[d] = depth_counts.get(d, 0) + 1
            limit, total = -1, 0
            for d in sorted(depth_counts):
                if total + depth_counts[d] > max_lines:
                    break
                total += depth_counts[d]
                limit = d
            if limit < 0:
                shown = rows[:max_lines]
                hidden_note = "showing the first %d of %d lines" % (max_lines, len(rows))
            else:
                shown = [(d, n) for d, n in rows if d <= limit]
                hidden_note = "showing call depth 0-%d (%d of %d lines)" % (limit, len(shown), len(rows))
        lines = []
        for d, n in shown:
            lines.append(line_for(n, d))
        if hidden_note:
            # Mark lines whose children were cut.
            cut_depth = max(d for d, _ in shown)
            for i, (d, n) in enumerate(shown):
                if d == cut_depth and n.children:
                    nested = len(n.children)
                    lines[i] += "  [+%d nested]" % nested
        out.append("calls (indent = called by the line above it; xN = times called; -> = return values seen):")
        out.extend(lines)
        if len(rows) >= MAX_RENDER:
            out.append("(listing stopped at %d lines)" % MAX_RENDER)
        if hidden_note:
            out.append("(%s; pass focus='<function or file regex>' to see the calls under a specific function)"
                       % hidden_note)
    with open(report, "w") as f:
        f.write("\n".join(out) + "\n")
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(exit_code if 0 <= exit_code < 256 else 1)


if __name__ == "__main__":
    main()
