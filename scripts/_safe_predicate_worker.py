#!/usr/bin/env python
"""Sandboxed executor for ONE candidate rule predicate. Stdlib only.

Reads {"code": <str>, "items": [{"text","label"}, ...]} as JSON on stdin; drops
privileges (setuid nobody) and sets RLIMIT_AS/RLIMIT_CPU; AST-allowlist-validates the
code (reject any node type not explicitly permitted, plus banned names / dunder
attrs / `format` / decorators / lambda / imports / f-strings); execs the def in a
restricted-`__builtins__` namespace; runs `rule(text)` over items; prints a JSON
result on stdout.
"""

import ast
import builtins
import json
import os
import sys

NOBODY = 65534

# Harmless leaf nodes (operators / contexts) are always allowed — they grant no
# capability. Everything else must be in ALLOWED_TYPES or the node is rejected.
_SAFE_BASES = (ast.operator, ast.cmpop, ast.boolop, ast.unaryop, ast.expr_context)
ALLOWED_TYPES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
    ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.Name, ast.Constant, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Call, ast.keyword, ast.Attribute, ast.Subscript, ast.Slice,
    ast.List, ast.Tuple, ast.Set, ast.Dict,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.IfExp, ast.Starred,
    ast.Try, ast.ExceptHandler, ast.Raise, ast.Assert,  # benign defensive code
}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "getattr", "setattr",
    "delattr", "hasattr", "vars", "dir", "globals", "locals", "type", "object",
    "super", "memoryview", "bytearray", "classmethod", "staticmethod", "property",
    "breakpoint", "help", "input", "__build_class__",
}
_SAFE_BUILTIN_NAMES = [
    "len", "str", "int", "bool", "float", "any", "all", "sum", "min", "max",
    "sorted", "set", "list", "dict", "tuple", "range", "enumerate", "zip",
    "isinstance", "abs", "ord", "chr", "reversed", "map", "filter", "round",
]


def drop_privileges() -> dict:
    info = {"dropped_uid": False, "rlimits": False}
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        info["rlimits"] = True
    except Exception as e:  # noqa: BLE001
        info["rlimit_error"] = repr(e)[:120]
    try:
        os.setgroups([])
        os.setgid(NOBODY)
        os.setuid(NOBODY)
        info["dropped_uid"] = True
    except Exception as e:  # noqa: BLE001
        info["uid_error"] = repr(e)[:120]
    sys.setrecursionlimit(2000)
    return info


def reject_reason(tree: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, _SAFE_BASES):
            continue
        if type(node) not in ALLOWED_TYPES:
            return f"node_type:{type(node).__name__}"
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            return "decorator"
        if isinstance(node, ast.Attribute) and ("__" in node.attr or node.attr == "format"):
            return f"attr:{node.attr}"
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"name:{node.id}"
    return None


def main() -> None:
    drop = drop_privileges()
    try:
        payload = json.loads(sys.stdin.read())
        code, items = payload["code"], payload["items"]
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "reason": "bad_input", "detail": repr(e)[:120], "drop": drop}))
        return
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(json.dumps({"ok": False, "reason": "syntax_error", "detail": str(e)[:120], "drop": drop}))
        return
    rej = reject_reason(tree)
    if rej:
        print(json.dumps({"ok": False, "reason": "ast_reject", "detail": rej, "drop": drop}))
        return
    safe = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(builtins, n)}
    safe.update({"True": True, "False": False, "None": None})
    ns = {"__builtins__": safe}
    try:
        exec(compile(tree, "<predicate>", "exec"), ns)  # noqa: S102 — sandboxed
        fn = ns.get("rule")
        if not callable(fn):
            print(json.dumps({"ok": False, "reason": "no_rule_fn", "drop": drop}))
            return
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "reason": "exec_def_error", "detail": repr(e)[:160], "drop": drop}))
        return
    correct = item_errors = 0
    for it in items:
        try:
            if bool(fn(it["text"])) == bool(it["label"]):
                correct += 1
        except Exception:  # noqa: BLE001  — a predicate that crashes on an item got it wrong
            item_errors += 1
    n = len(items)
    print(json.dumps({
        "ok": True, "accuracy": round(correct / n, 4) if n else None,
        "n": n, "n_item_errors": item_errors, "drop": drop,
    }))


if __name__ == "__main__":
    main()
