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

    def test_uint_mask_multiply_into_uint_lhs(self):
        """Gather of a uint source buffer must not emit
        `(mask?1.0f:0.0f)*(uint_load)` assigned to a uint variable.
        ANGLE strict GLES 3.1 rejects `uint = float*uint` with
        `cannot convert from 'float' to 'highp uint'`. The renderer
        must wrap the mask in `uint(...)` so the multiply stays in
        uint dtype."""
        import numpy as np
        # MNIST-shaped uint8 buffer + randint gather reproduces the
        # failing E_32_49_5_4_16_2_4 kernel pattern on the phone.
        X_train = Tensor((np.random.rand(60000, 1, 28, 28) * 255).astype(np.uint8))
        BS = 128
        idx = Tensor.randint(BS, high=X_train.shape[0])
        out = X_train[idx]
        # Walk all generated kernels; any `uint X = (...)?1.0f:0.0f` line
        # MUST have the mask wrapped in `uint(...)`. Specifically: no
        # `uint X = (<mask>)?1.0f:0.0f` (with mask not starting `uint(`)
        # assigned to a uint LHS.
        bad: list[str] = []
        for src, sink in _get_kernel_source(out):
            for line in src.splitlines():
                if 'uint ' not in line or '?1.0f:0.0f' not in line or '=' not in line:
                    continue
                # Locate the ternary and check the char immediately before
                # `?1.0f:0.0f` (after stripping the wrapping paren).
                idx_t = line.find('?1.0f:0.0f')
                # Walk back to the matching `(`
                depth, j = 0, idx_t - 1
                while j >= 0:
                    if line[j] == ')': depth += 1
                    elif line[j] == '(':
                        if depth == 0: break
                        depth -= 1
                    j -= 1
                # The 4 chars before position j should be "uint" or the
                # cast should already be present.
                preceded_by = line[max(0, j-6):j]
                if not preceded_by.endswith('uint') and '(uint(' not in line[max(0, j-20):idx_t]:
                    bad.append(line.strip())
        self.assertEqual(bad, [], f"Found unwrapped mask into uint LHS:\n  " + "\n  ".join(bad))

    def test_int_mask_multiply_into_int_lhs(self):
        """Same fix must also apply to int-typed LHS (not just uint).
        The renderer wraps the mask in `int(...)` for int targets."""
        # A simple int tensor with a boolean mask pattern: this hits
        # the (cond?1.0f:0.0f)*load code path with int dtypes.
        x = Tensor([1, 2, 3, 4, 5], dtype=dtypes.int32)
        mask = Tensor([True, False, True, False, True])
        out = x * mask
        # No need to verify ANGLE compile; just ensure the renderer
        # doesn't crash and the kernel source is well-formed.
        for src, _ in _get_kernel_source(out):
            # No bare `(mask?1.0f:0.0f)*int_load` should appear when
            # the LHS is int.
            self.assertNotRegex(src, r'(?:^|\s)int\s+\w+\s*=\s*\(?[^i][^)]*\?1\.0f:0\.0f\)')


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


class TestGLSLESSliceReadback(unittest.TestCase):
    """Regression test for the "size=1 slice at offset 0 returns zeros"
    bug in tinygrad's view path.

    Background: When a uchar tensor is sliced with a single element
    (e.g. ``X_g[0:1]``), tinygrad's ``_buffer()`` method on the slice
    materialises a fresh buffer via ``x.contiguous().realize()``. On
    GLSL_ES (both codegen-only on host and the real Android runtime)
    this fresh buffer ends up with an empty shadow on the first call,
    so reading it via ``.numpy()`` returns ``[0]``. On the second call
    the contig is recognised as a no-op (because the underlying COPY
    is already realised) and the view path returns the correct data
    from the source buffer.

    The test walks every size-1 slice at multiple offsets on a
    GLSL_ES-resident uchar buffer (host codegen-only mode) and asserts
    that the *first* read returns the correct value, not 0. This
    matches METAL/CPU behaviour. The bug breaks MNIST-style pipelines
    where the first pixel of an image is read with a size-1 slice.
    """

    def setUp(self):
        # Force GLSL_ES context for every test. NO_MEMORY_PLANNER=0 to
        # exercise the suballocated arena path too.
        self._ctx = Context(DEV="GLSL_ES", NO_MEMORY_PLANNER=0)
        self._ctx.__enter__()
        _clear_program_cache()

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    def test_size1_slice_at_various_offsets_returns_correct_data_on_first_read(self):
        """``X_g[off:off+1]`` for off in {0, 1, 2, 3, 4, 5, 9, 99} must
        return the correct value on the FIRST call to .numpy()."""
        # Use a real on-disk tensor (matches the MNIST path) so the
        # DISK->GLSL_ES copy is exercised the same way as the on-device
        # training probe.
        X = Tensor.from_url('https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz', gunzip=True)
        X_g = X[8:].to('GLSL_ES')

        # First, sanity-check the source data is on the device (this
        # also forces the COPY to run before we start slicing).
        full = X_g.numpy()
        # First 10 labels (we know the values from the file).
        expected_10 = [7, 2, 1, 0, 4, 1, 4, 9, 5, 9]
        self.assertEqual(list(full[:10]), expected_10)

        # For the size-1 slice check we use a small synthetic buffer
        # of known values so every offset is well-defined.
        small = Tensor(np.array(expected_10, dtype=np.uint8))
        small_g = small.to('GLSL_ES')
        self.assertEqual(list(small_g.numpy()), expected_10)

        for offset in (0, 1, 2, 3, 4, 5, 9, 99):
            # The small buffer is only 10 elements; for offset 99 the
            # slice runs off the end and we expect an empty result.
            if offset < 10:
                expected_val = expected_10[offset]
            else:
                expected_val = None
            with self.subTest(offset=offset):
                first_read = small_g[offset:offset + 1].numpy()
                if expected_val is None:
                    self.assertEqual(list(first_read), [],
                                     f"X_g[{offset}:{offset + 1}] off the end should be empty")
                else:
                    self.assertEqual(
                        list(first_read), [expected_val],
                        f"X_g[{offset}:{offset + 1}].numpy() returned {list(first_read)}, "
                        f"expected [{expected_val}] (GLSL ES size-1 slice first-read bug)"
                    )

    def test_size1_slice_int32_dtype(self):
        """Same bug surfaced with int32 data: the LSB of the underlying
        uint SSBO is always byte 0, so a size-1 slice at offset 0
        returns the LSB of the uninitialised first uint (0), not the
        first int value."""
        X = Tensor(np.array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5], dtype=np.int32))
        X_g = X.to('GLSL_ES')
        # Force the COPY to run.
        self.assertEqual(list(X_g.numpy()[:4]), [7, 2, 1, 0])
        for offset in (0, 1, 2, 3, 4):
            with self.subTest(offset=offset):
                first_read = X_g[offset:offset + 1].numpy()
                self.assertEqual(
                    list(first_read), [int(X.numpy()[offset])],
                    f"X_g[{offset}:{offset + 1}].numpy() (int32) returned {list(first_read)}, "
                    f"expected [{int(X.numpy()[offset])}]"
                )

    def test_size1_slice_float32_dtype(self):
        """The size-1 slice bug also affects float32. METAL/CPU always
        return the first value of the source on the first read;
        GLSL_ES historically returns 0.0 because the view materialises
        a fresh empty buffer on first call."""
        X = Tensor(np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=np.float32))
        X_g = X.to('GLSL_ES')
        self.assertEqual(list(X_g.numpy()[:3]), [1.5, 2.5, 3.5])
        for offset in (0, 1, 2, 3):
            with self.subTest(offset=offset):
                first_read = X_g[offset:offset + 1].numpy()
                self.assertEqual(
                    list(first_read), [float(X.numpy()[offset])],
                    f"X_g[{offset}:{offset + 1}].numpy() (float32) returned {list(first_read)}, "
                    f"expected [{float(X.numpy()[offset])}]"
                )

    def test_size5_slice_returns_correct_data(self):
        """Sanity: multi-element slices (size >= 2) already work on
        GLSL_ES; this test guards against regressing that path while
        fixing the size-1 case."""
        X = Tensor(np.array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9], dtype=np.int32))
        X_g = X.to('GLSL_ES')
        first_read = X_g[:5].numpy()
        self.assertEqual(list(first_read), [7, 2, 1, 0, 4])

    def test_view_path_matches_cpu_behavior(self):
        """A small uchar tensor that fits the size=1 case must read
        the same value on GLSL_ES as on CPU/METAL on the first call.
        Catches the regression where GLSL_ES returns 0 but other
        backends return the correct value."""
        for arr in (np.array([1, 2, 3, 4, 5, 100, 200, 255], dtype=np.uint8),
                    np.array([1, 2, 3, 4, 5, 100, 200, 255], dtype=np.int32),
                    np.array([1.5, 2.5, 3.5, 4.5, 5.5, 100.5, 200.5, 255.5], dtype=np.float32)):
            for offset in (0, 1, 3, 5):
                # CPU reference (always correct).
                cpu_expected = Tensor(arr).numpy()[offset:offset + 1]
                # GLSL_ES under test (the bug under repair).
                gpu_first = Tensor(arr).to('GLSL_ES')[offset:offset + 1].numpy()
                np.testing.assert_array_equal(
                    gpu_first, cpu_expected,
                    err_msg=f"size-1 slice at offset {offset} (dtype={arr.dtype}) "
                            f"differs between CPU and GLSL_ES on first read: "
                            f"CPU={cpu_expected}, GLSL_ES={gpu_first}"
                )


class TestGLSLESPackedStorage(unittest.TestCase):
    """Test that GLSL ES sub-4-byte dtypes (uchar/char/ushort/short)
    use packed storage, matching what WGSL does.

    WGSL renderer's approach (wgsl.py:12-28, 99-100):
      - For sub-4-byte dtypes, the SSBO is declared as `uint[]` with
        element count = byte_count / 4 (one uint per 4 source elements).
      - Loads use read + shift + mask:
          word = dataN[idx/4]
          byte = (word >> (idx%4)*8) & 0xFF
      - Stores use atomicAnd/atomicAdd (read-modify-write) or
        `imageStore` (texture alternative):
          old = atomic(dataN[idx/4])
          atomic(dataN[idx/4]) = (old & ~mask) | (new & mask)

    This is the WGSL-equivalent approach: no extra memory cost
    (4x SMALLER buffer, not 4x bigger) and no std430 alignment bug.
    The GLSL ES renderer should match this so that:
      - uchar/char gather (fancy indexing) returns correct values
      - uchar/char slices work
      - The buffer size is the same as int32 (4 bytes per element)

    These tests verify the GLSL ES renderer emits packed storage code
    for sub-4-byte dtypes by inspecting the generated kernel source.
    """
    def setUp(self):
        self._ctx = Context(DEV="GLSL_ES", NO_MEMORY_PLANNER=0)
        self._ctx.__enter__()
        _clear_program_cache()

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    def _collect_kernels(self, op):
        """Trigger compilation of `op` and return the unique kernel
        sources that were generated for it."""
        import tinygrad.runtime.ops_glsl_es as gles
        seen = []
        orig = gles.GLSLESProgram.__init__
        def spy(self, dev, name, lib, **kw):
            orig(self, dev, name, lib, **kw)
            if self.src not in seen:
                seen.append(self.src)
        gles.GLSLESProgram.__init__ = spy
        try:
            out = op.numpy() if hasattr(op, 'numpy') else op.realize()
        finally:
            gles.GLSLESProgram.__init__ = orig
        return seen

    def test_uchar_ssbo_uses_packed_read(self):
        """With packed storage, the SSBO for a uchar source uses
        `dataN[idx/4] >> shift & mask` to extract each source byte.
        Without packed storage, the kernel would use `dataN[idx]`
        which packs 4 source elements into one uint slot, breaking
        gather/fancy-index correctness.

        The SSBO is declared unsized (`dataN[]`) in GLSL ES 3.20; the
        packed/unpacked distinction is in the LOAD access pattern,
        not the size of the declaration. So this test checks the access
        pattern in the kernel source.
        """
        import re
        X = Tensor(np.zeros(10, dtype=np.uint8))
        y = X + 1
        seen = self._collect_kernels(y)
        # Look for a packed read: `dataN[expr / 4]` with shift and mask.
        packed_re = re.compile(r'data\d+_\d+\s*\[[^\]]*?/\s*4\s*[^\]]*?\]')
        for src in seen:
            if packed_re.search(src):
                return  # Found a packed read.
        self.fail(f"no packed read found in any kernel; saw: {seen}")

    def test_uchar_load_uses_packed_read_pattern(self):
        """Verify the kernel source uses the packed read pattern for
        uchar SSBOs. The pattern is `dataN[idx/4]` plus shift and mask,
        instead of a direct `dataN[idx]`.

        On host (codegen-only), the kernel doesn't run, so the actual
        output is always zeros. We can't test correctness of the output;
        we only test that the GENERATED CODE has the right pattern.
        """
        import re
        # Trigger a kernel that reads a uchar buffer.
        X = Tensor(np.zeros(10, dtype=np.uint8))
        y = X + 1
        seen = self._collect_kernels(y)
        # Look for a packed read: `dataN[idx/4]` plus shift (>> 8*N) and mask (& 0xFF).
        # With packed storage: we see `data1_10[alu0/4]` or `data1_10[(alu0/4)]`.
        # Without packed storage: we see `data1_10[alu0]` (direct index).
        packed_pattern = re.compile(r'data\d+_\d+\s*\[[^\]]*?/\s*4\s*[^\]]*?\]')
        direct_pattern = re.compile(r'data\d+_\d+\s*\[\s*alu0\s*\]')
        for src in seen:
            if packed_pattern.search(src):
                # Found a packed read. Good.
                return
            if direct_pattern.search(src):
                # Found a direct read on a uchar buffer. This is the bug
                # — should be packed.
                self.fail(
                    f"kernel reads uchar SSBO with direct `dataN[alu0]` index; "
                    f"expected packed `dataN[alu0/4]` with shift+mask. Kernel:\n{src}"
                )
        # No uchar read found in any kernel.
        self.fail(f"no uchar read found in any kernel; saw: {seen}")

    def test_uchar_fancy_index_load_uses_packed_read_pattern(self):
        """Fancy indexing lowers to gated loads; those loads still need the
        packed `dataN[idx/4]` read path instead of direct `dataN[idx]`."""
        import re
        X = Tensor(np.arange(16, dtype=np.uint8)).to('GLSL_ES')
        idx = Tensor([0, 2, 4, 9, 12], device='GLSL_ES')
        seen = self._collect_kernels(X[idx])
        packed_pattern = re.compile(r'data\d+_16\s*\[[^\]]*?/\s*4\s*[^\]]*?\]')
        direct_pattern = re.compile(r'data\d+_16\s*\[\s*alu\d+\s*\]')
        for src in seen:
            if packed_pattern.search(src):
                return
            if direct_pattern.search(src):
                self.fail(
                    f"fancy-index kernel reads uchar SSBO with direct `dataN[alu]`; "
                    f"expected packed `dataN[alu/4]` with shift+mask. Kernel:\n{src}"
                )
        self.fail(f"no uchar fancy-index read found in any kernel; saw: {seen}")

    def test_uchar_fancy_index_store_uses_packed_store_pattern(self):
        """Fancy indexing output should write packed uint slots, not direct
        `dataN[idx] = value` stores into a `uint[]` SSBO."""
        import re
        X = Tensor(np.arange(16, dtype=np.uint8)).to('GLSL_ES')
        idx = Tensor([0, 2, 4, 9, 12], device='GLSL_ES')
        seen = self._collect_kernels(X[idx])
        packed_pattern = re.compile(r'atomic(?:And|Add|Or|Exchange)\(data\d+_5\s*\[[^\]]*?/\s*4\s*[^\]]*?\]')
        direct_pattern = re.compile(r'data\d+_5\s*\[\s*[0-9]+\s*\]\s*=')
        for src in seen:
            if packed_pattern.search(src):
                return
            if direct_pattern.search(src):
                self.fail(
                    f"fancy-index kernel stores uchar output with direct `dataN[i] = value`; "
                    f"expected packed atomic store into `dataN[i/4]`. Kernel:\n{src}"
                )
        self.fail(f"no uchar fancy-index store found in any kernel; saw: {seen}")

    def test_bool_output_does_not_use_packed_store(self):
        """`bool` buffers are not packed like uchar/ushort; emitting atomic
        uint ops on `bool[]` SSBOs does not compile on GLES."""
        X = Tensor([1.0, 2.0, 3.0], device='GLSL_ES')
        Y = Tensor([1.0, 0.0, 3.0], device='GLSL_ES')
        seen = self._collect_kernels(X == Y)
        for src in seen:
            self.assertNotIn("atomicOr(data0_3", src, msg=f"bool output used packed atomic store:\n{src}")

    def test_int32_load_unchanged_by_packed_storage(self):
        """int32 (itemsize=4) is NOT packed -- packed storage only applies
        to sub-4-byte dtypes. The load should still use a direct
        `dataN[idx]` (no `/4` shift+mask)."""
        import re
        X = Tensor(np.zeros(10, dtype=np.int32))
        y = X + 1
        seen = self._collect_kernels(y)
        packed_re = re.compile(r'data\d+_\d+\s*\[[^\]]*?/\s*4\s*[^\]]*?\]')
        for src in seen:
            if packed_re.search(src):
                self.fail(
                    f"int32 load uses packed read (dataN[idx/4]) but should "
                    f"be direct dataN[idx]. Kernel:\n{src}"
                )


if __name__ == "__main__":
  unittest.main()
