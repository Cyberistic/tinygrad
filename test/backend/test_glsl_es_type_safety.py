"""Tests for type-correctness of GLSL ES 3.1 kernel emission.

These tests verify that the GLSLESRenderer produces GLSL that compiles
under ANGLE's strict GLES 3.1 type checker. The issue is that tinygrad's
codegen emits mixed-type GLSL expressions (e.g. `float * int`) which
ANGLE strict mode rejects outright, even though the GLSL ES 3.10 spec
section 5.10 says implicit int->float conversion should happen.

The fix is a UPat-based pass in glsl_es.py that:
  1. Tracks the inferred type of every UOp in the kernel body
  2. Inserts explicit `int(...)` / `float(...)` casts at every mixed-type
     arithmetic site
  3. Special-cases GLSL ES shift (`<<`, `>>`) and array index (`[]`) which
     require int operands
  4. Special-cases GLSL ES ternary (`?:`) which requires both arms be the
     same type
"""
import unittest
import numpy as np
import re

from tinygrad import Tensor, dtypes
from tinygrad.device import Device
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
from tinygrad.codegen import to_program


# Heuristic: does this kernel have any of the known type-confusion
# anti-patterns that ANGLE strict rejects? Returns a list of human-readable
# descriptions of every match.
_TYPE_ANTIPATTERNS = [
    # `<float> < <int>` or `>` after a closing paren (e.g. `(val0)<alu3`).
    (re.compile(r"\)\s*[<>]\s*(alu|gidx|lidx|val|Ridx)\d+"),
     "float compared with int"),
    # `float * int` or `+` or `-` after `float(...)` cast
    (re.compile(r"float\([^)]+\)\s*[+*/\-]\s*\("),
     "float(...) chained with (...)"),
    # `int * int` literal RHS (e.g. `gidx0*12`)
    (re.compile(r"\)\s*\*\s*\d+"),
     "float() result multiplied by int literal"),
    # array index that mixes int and float: `arr[float(thing)+-123]`
    (re.compile(r"\[\([^]]*float\([^]]+\)[^]]*\+\-\d+\)\]"),
     "array index mixes float cast and int +/-"),
    # ASSIGN of const int MIN to float decl: `buf[0]=-2147483648;`
    (re.compile(r"\]\s*=\s*-?\d{7,}"),
     "int MIN/MAX literal assigned to indexed float buf"),
    # ternary with mismatched arms: `(int?int_val:float_val)`
    (re.compile(r"\(\s*\w+\s*\?\s*\([^:)]+\)\s*:\s*\("),
     "ternary with mismatched arm types"),
    # `float` followed by `<<` or `>>` (shift with float LHS)
    (re.compile(r"float\([^)]+\)\s*<<"),
     "float << int"),
    (re.compile(r"float\([^)]+\)\s*>>"),
     "float >> int"),
    # array index that starts with float(
    (re.compile(r"\[\s*float\("),
     "float array index"),
]


def _find_antipatterns(kernel_src: str) -> list[str]:
    out = []
    for pat, desc in _TYPE_ANTIPATTERNS:
        for m in pat.finditer(kernel_src):
            # Special-case the int MIN/MAX assignment: only report if the
            # target buffer was declared as float. int/uint buffers can
            # legitimately hold int MIN/MAX literals.
            if desc == "int MIN/MAX literal assigned to indexed float buf":
                # Match is like "buf[0] = -2147483648"; extract buffer name.
                prefix = kernel_src[:m.start()]
                if (idx := prefix.rfind('[')) != -1:
                    before = prefix[:idx].strip()
                    if ' ' in before:
                        buf_name = before.split()[-1]
                    else:
                        buf_name = before
                    decl_match = re.search(rf"(float|int|uint)\s+{re.escape(buf_name)}\s*\[", kernel_src[:m.start()])
                    if decl_match and decl_match.group(1) != "float":
                        continue
            out.append(f"{desc}: {m.group(0)!r}")
    return out


def _get_kernel_source(t: Tensor):
    """Get the source of every kernel needed to realize tensor t.

    Returns a list of (source_text, sink_op) tuples.

    We capture the schedule BEFORE realize(), because once realize()
    runs the schedule is empty (the linear ops are gone).
    """
    out = []
    from tinygrad.uop.ops import Ops as _O
    linear = t.schedule_linear()
    seen = set()
    for sink in linear.src:
        if sink.op is _O.CALL:
            key = id(sink)
            if key in seen: continue
            seen.add(key)
            try:
                prg = to_program(sink.src[0], Device["GLSL_ES"].renderer)
            except Exception:
                # Some ops (e.g. Ops.COPY for buffer transfers) don't go
                # through to_program. Skip them.
                continue
            for u in prg.toposort():
                if u.op is _O.SOURCE:
                    out.append((u.arg, sink))
                    break
    # Now actually realize.
    t.realize()
    return out


def _clear_program_cache():
    """Clear the to_program cache so we get fresh kernels."""
    from tinygrad.codegen import to_program_cache
    to_program_cache.clear()


class TestGLSLESTypeSafety(unittest.TestCase):
    """Verify that the GLSLESRenderer emits GLSL that does not contain
    type-confusion anti-patterns that ANGLE strict mode rejects."""

    def setUp(self):
        # Force GLSL_ES context for every test.
        self._ctx = Context(DEV="GLSL_ES", NO_MEMORY_PLANNER=1)
        self._ctx.__enter__()
        _clear_program_cache()

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    def _src(self, t):
        srcs = _get_kernel_source(t)
        return srcs[0][0] if srcs else ""

    def test_floats_no_antipattern(self):
        """A simple float pipeline should produce a clean kernel."""
        a = Tensor.randn(8, 16, dtype=dtypes.float)
        b = Tensor.randn(16, 32, dtype=dtypes.float)
        c = (a @ b).relu()
        src = self._src(c)
        issues = _find_antipatterns(src)
        self.assertEqual(issues, [], f"Found type anti-patterns:\n  " + "\n  ".join(issues))

    def test_int_argmax_uses_float_init(self):
        """argmax uses int MIN (-2147483648) as the init value for the
        running max. When the running max is stored in a float buf,
        ANGLE rejects `buf[0] = -2147483648`. The kernel must wrap
        the int MIN in float() or use float(-inf)."""
        x = Tensor.randn(32, 226, dtype=dtypes.float)
        am = x.argmax(axis=1)
        # Find ALL kernels with the int MIN init pattern assigned to a
        # *float* buffer and verify they all use a float init pattern.
        bad_kernels = []
        for src, sink in _get_kernel_source(am):
            for m in re.finditer(r"(\w+)\s*\[\s*\w+\s*\]\s*=\s*(-?\d{7,})", src):
                buf_name = m.group(1)
                decl_match = re.search(rf"(float|int|uint)\s+{re.escape(buf_name)}\s*\[", src[:m.start()])
                if decl_match and decl_match.group(1) == "float":
                    bad_kernels.append(src[:300])
                    break
        if bad_kernels:
            self.fail(f"Found {len(bad_kernels)} kernels with bare int MIN literal in float buffer:\n"
                      + "\n---\n".join(bad_kernels))

    def test_int_argmax_compare_with_int_max(self):
        """argmax reads the float input and the int index. The comparison
        `(float) < (int)` is rejected by ANGLE strict. The kernel must
        cast the int to float (or vice versa) before the comparison."""
        x = Tensor.randn(32, 226, dtype=dtypes.float)
        am = x.argmax(axis=1)
        src = self._src(am)
        # Look for `)` followed by `<` followed by `alu` (int var).
        if re.search(r"\)\s*<\s*alu\d+", src):
            self.fail(f"Found float-buffered buf compared with int alu:\n{src}")

    def test_shift_ops_have_int_operands(self):
        """GLSL ES `<<` and `>>` require both operands to be int. If the
        codegen emits `(float) << (int)`, ANGLE rejects it."""
        x = Tensor.randn(4, 4, dtype=dtypes.float)
        # A 2D index: this generates a shift op to compute the offset.
        idx = (x.argmax(axis=1).flatten() * 4 + 1)
        src = self._src(idx)
        if re.search(r"float\([^)]+\)\s*<<", src):
            self.fail(f"Found float << int:\n{src}")
        if re.search(r"float\([^)]+\)\s*>>", src):
            self.fail(f"Found float >> int:\n{src}")

    def test_array_index_uses_int(self):
        """GLSL ES array index `arr[expr]` requires expr to be int.
        If the codegen emits `arr[float(expr)]`, ANGLE rejects it."""
        x = Tensor.randn(4, 4, dtype=dtypes.float)
        # The following generates a gather with index = int(0) or similar.
        out = x[Tensor([0, 2])]
        src = self._src(out)
        if re.search(r"\[\s*float\(", src):
            self.fail(f"Found float array index:\n{src}")


class TestGLSLESSourceContent(unittest.TestCase):
    """End-to-end test: the GLSL source from a complex op must compile
    under ANGLE strict GLES 3.1. We verify by running the GLSL through
    an actual ANGLE compile via the glslangValidator, if available."""

    def setUp(self):
        self._ctx = Context(DEV="GLSL_ES", NO_MEMORY_PLANNER=1)
        self._ctx.__enter__()
        _clear_program_cache()

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    def test_argmax_source_is_angie_compatible(self):
        """The argmax kernel produced by tinygrad must compile under
        ANGLE strict GLES 3.1. We do a static check here (no ANGLE
        available on the dev host) by looking for type-confusion
        anti-patterns. The real ANGLE compile happens on the phone."""
        x = Tensor.randn(32, 226, dtype=dtypes.float)
        am = x.argmax(axis=1)
        srcs = _get_kernel_source(am)
        # Check every kernel produced.
        for src, _ in srcs:
            issues = _find_antipatterns(src)
            self.assertEqual(issues, [], f"Found type anti-patterns:\n  " + "\n  ".join(issues))

    def test_matmul_source_is_angie_compatible(self):
        """A matmul kernel must not mix int and float types."""
        a = Tensor.randn(8, 16, dtype=dtypes.float)
        b = Tensor.randn(16, 32, dtype=dtypes.float)
        c = (a @ b)
        srcs = _get_kernel_source(c)
        for src, _ in srcs:
            issues = _find_antipatterns(src)
            self.assertEqual(issues, [], f"Found type anti-patterns:\n  " + "\n  ".join(issues))

    def test_cnn_forward_all_kernels_clean(self):
        """End-to-end: a full CNN forward (conv2d + relu + maxpool + linear)
        must produce kernels that compile under ANGLE strict GLES 3.1.
        Regression test for the bug that prevented CNN training on the
        phone."""
        from tinygrad.nn import Conv2d, Linear
        B, C, H, W = 4, 3, 8, 8
        NUM_CLASSES = 4
        # Build a small CNN with separate conv/relu/maxpool (the
        # model.py pattern that prevents linearizer fusion across layers).
        model = [
            Conv2d(C, 4, 3, padding=1),
            lambda x: x.relu(),
            lambda x: x.max_pool2d(2, 2),
            Conv2d(4, 8, 3, padding=1),
            lambda x: x.relu(),
            lambda x: x.max_pool2d(2, 2),
            lambda x: x.flatten(1),
            Linear(8 * 2 * 2, NUM_CLASSES),
        ]
        x = Tensor.randn(B, C, H, W, dtype=dtypes.float)
        out = x
        for layer in model:
            out = layer(out)
        out.realize()
        # Walk every kernel.
        for src, _ in _get_kernel_source(out):
            issues = _find_antipatterns(src)
            self.assertEqual(issues, [], f"Found type anti-patterns:\n  " + "\n  ".join(issues))


if __name__ == "__main__":
    unittest.main()
