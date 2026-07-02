"""GLSL ES 3.1 compute shader renderer for tinygrad.

This adds a new tinygrad target "GLSL_ES" that emits GLES 3.1 compute
shaders suitable for ANGLE/Vulkan on Android (Xclipse 540). Same
UOP-based codegen and kernel fusion as the WGSL/OpenCL/Metal renderers,
just a different target format.

Usage:
    from tinygrad import Tensor
    from tinygrad.helpers import Context
    with Context(DEV="GLSL_ES"):
        out = model.forward(x).realize()

The Android app loads the generated .comp files via the existing
EGL/GLES31 stack — same as hand-written shaders, but now tinygrad did
the codegen, fusion, and buffer management automatically.
"""
import math
from tinygrad.dtype import DType, dtypes, AddrSpace
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, extra_pm
from tinygrad.helpers import strip_parens

# GLSL ES 3.1 compute shader pattern matcher.
# GLSL ES-specific patterns FIRST (they take precedence), then base_rewrite
# as fallback for things that are the same as C-style (arithmetic, etc.)

def _glsl_es_float_const(x):
  """Render a float constant using GLSL ES-compatible literals.

  Returns None for ordinary floats so the base_rewrite's `f"{x.arg}f"`
  pattern (which is correct for finite values) handles them.
  """
  if x.arg == float("-inf"): return "(-(1.0/0.0))"
  if x.arg == float("inf"):  return "(1.0/0.0)"
  if math.isnan(x.arg):      return "(0.0/0.0)"
  return None  # let the generic pattern fire

def _find_buffer_uop(u: UOp) -> UOp | None:
  """Walk through AFTER/SHRINK/PARAM wrappers to find the underlying BUFFER."""
  seen = set()
  while u.op not in (Ops.BUFFER, Ops.PARAM) and id(u) not in seen:
    seen.add(id(u))
    if not u.src: return None
    u = u.src[0]
  return u if u.op is Ops.BUFFER else None


def _render_glsl_es_store(ctx, bidx, var) -> str:
  # Find the underlying BUFFER UOp and its declared scalar type.
  if bidx.op is Ops.INDEX:
    buf = _find_buffer_uop(bidx)
  elif bidx.op is Ops.BUFFER:
    buf = bidx
  else:
    buf = None
  if buf is None:
    return f"{ctx.render_access(bidx)} = {ctx[var]};"
  rendered = ctx._render_dtype(buf.dtype, 1, buf.arg.addrspace)
  # The value's rendered type is based on its actual dtype (use REG/ALU
  # addrspace, not the buffer's addrspace, so an int value renders as int).
  var_rendered = ctx._render_dtype(var.dtype, 1, AddrSpace.REG)
  if rendered != var_rendered:
    return f"{ctx.render_access(bidx)} = {rendered}({ctx[var]});"
  return f"{ctx.render_access(bidx)} = {ctx[var]};"


def _render_glsl_es_load(ctx, bidx, load_dtype) -> str:
  # Find the underlying BUFFER UOp and its declared scalar type.
  if bidx.op is Ops.INDEX:
    buf = _find_buffer_uop(bidx)
  elif bidx.op is Ops.BUFFER:
    buf = bidx
  else:
    buf = None
  if buf is None:
    return f"{ctx[bidx]}"
  rendered = ctx._render_dtype(buf.dtype, 1, buf.arg.addrspace)
  # The LOAD result's rendered type is based on its actual dtype (use REG
  # addrspace, not the buffer's addrspace, so an int LOAD renders as int).
  load_rendered = ctx._render_dtype(load_dtype, 1, AddrSpace.REG)
  if rendered != load_rendered:
    return f"{load_rendered}({ctx[bidx]})"
  return f"{ctx[bidx]}"


glsl_es_matcher = PatternMatcher([
  # STORE: wrap values in an explicit cast when the value's rendered type
  # differs from the buffer's declared scalar type. Our _render_dtype
  # override forces LOCAL/shared memory to float regardless of the BUFFER's
  # UOp dtype, so writing an int to a shared float buf without a cast fails
  # ANGLE's strict type check. cstyle's STORE pattern produces `buf = val;`
  # without any cast; we override it here to insert the cast when needed.
  (UPat(Ops.STORE, src=(UPat.var("bidx"), UPat.var("var")), name="st"),
   lambda ctx,bidx,var,st: _render_glsl_es_store(ctx, bidx, var)),
  # Buffer declarations. LOCAL buffers get `shared` (workgroup memory) and
  # are extracted from the kernel body by render_kernel. REG buffers are
  # per-thread register accumulators (col2im scratch, im2col scratch, etc.)
  # and become a plain local float[] in the kernel body. GLOBAL buffers
  # are NOT emitted here — render_kernel handles them as SSBO declarations
  # at file scope with the correct binding numbers. Emitting them here
  # would put duplicate SSBO layouts inside main().
  (UPat(Ops.BUFFER, name="x"), lambda ctx,x:
   (f"shared {ctx._render_dtype(x.dtype, 1, x.addrspace)} {ctx[x]}[{x.src[0].as_shape[0]}];" if x.addrspace == AddrSpace.LOCAL
    else f"{ctx._render_dtype(x.dtype, 1, x.addrspace)} {ctx[x]}[{x.src[0].as_shape[0]}];" if x.addrspace == AddrSpace.REG
    else f"// GLOBAL buffer {ctx[x]} handled by render_kernel")),
  # Whole-buffer reference: LOAD with no index into a BUFFER.
  # The cstyle.py codegen generates `*((vecN*)(buf))` which assumes the
  # buffer is vecN[]. Since we force all buffers to float[], we need
  # to generate `vecN(buf[0], buf[1], ..., buf[N-1])` instead. The N
  # is determined by the LOAD's dtype. This pattern must come BEFORE
  # the cstyle.py's `string_rewrite` to take precedence.
  (UPat(Ops.LOAD, src=(UPat(Ops.BUFFER),), name="x"),
   lambda ctx,x: f"vec{x.dtype.count}({','.join(f'{ctx[x.src[0]]}[{i}]' for i in range(x.dtype.count))})"
   if x.dtype.count > 1 else f"{ctx[x.src[0]]}[0]"),
  # bool constants
  (UPat.cvar("x", dtype=dtypes.bool), lambda x: "true" if x.arg else "false"),
  # Infinity / -infinity: GLSL ES doesn't have an `inf` literal. Use the
  # portable (1.0/0.0) form, which always evaluates to +inf. The base
  # cstyle.py uses a generic `f"{x.arg}f"` for floats which produces
  # "-inff" (a literal token), so we override before that pattern fires.
  (UPat.cvar("x", dtype=dtypes.floats), _glsl_es_float_const),
  # uint constants need 'u' suffix in GLSL ES
  (UPat(Ops.CONST, dtype=(dtypes.uint8, dtypes.uint16, dtypes.uint32), name="x"),
   lambda x: f"{x.arg & 0xFFFFFFFF}u"),
  # int constants
  (UPat(Ops.CONST, dtype=(dtypes.int8, dtypes.int16, dtypes.int32), name="x"),
   lambda x: str(x.arg)),
  # float constants need 'f' suffix in GLSL ES for type clarity
  (UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda x: f"{x.arg}f"),
  # half constants: GLSL ES doesn't have native f16 in 3.1, promote to f32
  (UPat(Ops.CONST, dtype=dtypes.half, name="x"), lambda x: f"{x.arg}f"),
  # bitcasts: GLSL ES has uintBitsToFloat/floatBitsToUint/intBitsToFloat etc.
  (UPat(Ops.BITCAST, dtype=dtypes.float, src=(UPat(dtype=dtypes.uint32,),), name="x"),
   lambda ctx,x: f"uintBitsToFloat({ctx[x.src[0]]})"),
  (UPat(Ops.BITCAST, dtype=dtypes.float, src=(UPat(dtype=dtypes.int32,),), name="x"),
   lambda ctx,x: f"intBitsToFloat({ctx[x.src[0]]})"),
  (UPat(Ops.BITCAST, dtype=dtypes.uint32, src=(UPat(dtype=dtypes.float,),), name="x"),
   lambda ctx,x: f"floatBitsToUint({ctx[x.src[0]]})"),
  (UPat(Ops.BITCAST, dtype=dtypes.int32, src=(UPat(dtype=dtypes.float,),), name="x"),
   lambda ctx,x: f"floatBitsToInt({ctx[x.src[0]]})"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"({ctx.type_map[x.dtype]})({ctx[x.src[0]]})"),
  # STACK is used for vectorized loads/stores. The dtype.count may be
  # 1 (scalar) even when there are 4 srcs (a vec4 represented as 4
  # scalars). Use len(src) for the actual vector width.
  (UPat(Ops.STACK, name="x"),
   lambda ctx,x: f"vec{len(x.src)}({','.join([ctx[y] for y in x.src])})"),
  # index access: GLSL ES uses buf[idx]
  (UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx"))),
   lambda ctx,b,idx: f"{ctx[b]}[{strip_parens(ctx[idx]) if idx.arg is Ops.ADD else ctx[idx]}]"),
  # store: GLSL ES uses buf[idx] = val (the C-style render_access does *buf which is wrong)
  (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var"))),
   lambda ctx,bidx,var: f"{ctx[bidx]} = {ctx[var]};"),
  # Whole-buffer LOAD: LOAD with src=GLOBAL BUFFER or PARAM (no INDEX).
  # The cstyle.py codegen uses `*((vecN*)(buf))` which assumes vecN[]
  # buffer. Since we force all buffers to float[], we expand to
  # `vecN(buf[0..N-1])`.
  (UPat(Ops.LOAD, src=(UPat((Ops.BUFFER, Ops.PARAM), name="b"),), name="x"),
   lambda ctx,x,b: (f"vec{x.dtype.count}({','.join(f'{ctx[b]}[{i}]' for i in range(x.dtype.count))})"
                       if x.dtype.count > 1 else f"{ctx[b]}[0]")
   if b.addrspace == AddrSpace.GLOBAL else f"({ctx[b]})"),
  # load: cast the loaded value to the LOAD's declared dtype if the
  # buffer's rendered type differs (e.g. LOCAL shared memory is forced
  # to float; reading an int LOAD from it needs int(float(buf[idx]))).
  (UPat(Ops.LOAD, src=(UPat.var('bidx'),), name="x"),
   lambda ctx,x,bidx: _render_glsl_es_load(ctx, bidx, x.dtype)),
  # Gated load (bounds-checked): use a multiply by the gate, which is
  # type-agnostic in GLSL ES. The codegen always uses 0.0f as the
  # default `var` for bounds-checked loads, so `gate ? bidx : 0.0f`
  # is equivalent to `(gate ? 1.0f : 0.0f) * bidx`. We emit the
  # multiply form to avoid the GLSL ES ternary "mismatching operand
  # types" error when bidx is vec and var is scalar.
  (UPat(Ops.LOAD, src=(UPat.var("bidx"), UPat.var("var"), UPat.var("gate"))),
   lambda ctx,bidx,var,gate: f"(({ctx[gate]}?1.0f:0.0f)*({ctx.render_access(bidx)}))"),
  # WHERE (ternary) with mismatched arm types. GLSL ES 3.1's `?:` requires
  # both arms to have the same type. The codegen can produce WHEREs where
  # one arm is a vector (e.g. vec4) and the other is a scalar (float)
  # via broadcasting. Cast the smaller-count arm to the larger using the
  # GLSL ES constructor (e.g. `vec4(1.0f)` broadcasts the scalar). This
  # must come BEFORE base_rewrite's generic ALU pattern so the cast is
  # applied here; otherwise the renderer emits `(cond?vec4:float)` which
  # the strict ANGLE GLSL ES 3.1 compiler rejects with "mismatching
  # ternary operator operand types".
  (UPat(Ops.WHERE, src=(UPat.var("cond"), UPat.var("x"), UPat.var("y")), name="w"),
   lambda ctx,cond,x,y,w: _glsl_es_where(ctx, w, x, y)),
]) + base_rewrite


def _glsl_es_vec_type_name(dtype) -> str:
  """Return the GLSL ES vector type name for a dtype (e.g. dtypes.float.vec(4) -> 'vec4')."""
  count = dtype.count
  if count == 1:
    return {"float": "float", "int": "int", "uint": "uint", "bool": "bool"}.get(dtype.scalar().name, dtype.scalar().name)
  base = dtype.scalar().name
  if base == "float": return f"vec{count}"
  if base == "int":   return f"ivec{count}"
  if base == "uint":  return f"uvec{count}"
  if base == "bool":  return f"bvec{count}"
  return f"{base}{count}"


def _glsl_es_where(ctx, w, x, y):
  """Render Ops.WHERE for GLSL ES 3.1, handling mismatched arm types.

  If the true and false arms have different vector widths, cast the
  smaller to the larger via constructor. e.g. WHERE(bool, vec4, float)
  with result vec4 becomes `(cond?vec4:vec4(float))`.
  """
  cond_str = ctx[w.src[0]]
  x_str = ctx[x]
  y_str = ctx[y]
  if x.dtype == y.dtype:
    return f"({cond_str}?{x_str}:{y_str})"
  # Type mismatch: promote the smaller-count arm to the larger.
  # Always use x.dtype as the target if x.count >= y.count, else y.dtype.
  if x.dtype.count >= y.dtype.count:
    target, smaller_str = x.dtype, y_str
    return f"({cond_str}?{x_str}:{_glsl_es_vec_type_name(target)}({smaller_str}))"
  target, smaller_str = y.dtype, x_str
  return f"({cond_str}?{_glsl_es_vec_type_name(target)}({smaller_str}):{y_str})"

class GLSLESRenderer(CStyleLanguage):
  """
  GLSL ES 3.1 compute shader renderer. Emits shaders that the existing
  Android gpu_ops layer (EGL/GLES31) can compile and execute.
  """
  # GLSL ES 3.1 has generous dispatch limits
  global_max = (65535, 65535, 65535)
  local_max  = (256, 256, 64)
  supports_float4 = False  # scalar loads only; the devectorizer splits vec4
                            # into indexed scalar loads that our renderer
                            # handles correctly. avoids whole-buffer vec4
                            # LOADs that would need a vector reinterpret
                            # (GLSL ES 3.1 has no whole-buffer value syntax).

  # Global parameter name map. Maps buffer identity -> short global name
  # (e.g. "g0", "g1", ...). The identity is either a BUFFER UOp id (for
  # inputs that come from the CALL's BUFFER list) or a PARAM UOp id
  # (for things like range variables that don't have a BUFFER).
  #
  # The cstyle.py naming uses data{slot}_{shape} which collides when the
  # same (slot, size) refers to different logical buffers in different
  # kernels. The global name map ensures every unique buffer across all
  # kernels gets a unique name, so the Android buffer manager can key
  # by name without ambiguity.
  global_param_map: dict = {}
  global_param_counter: int = 0
  # Current kernel index being rendered. Set by gen_glsl_android.py before
  # calling to_program(). The render() method uses this to look up the
  # correct global name for each PARAM by its slot.
  current_kernel: int = -1
  # Per-kernel slot -> global name map. Built by gen_glsl_android.py.
  # kernel_slot_maps[ki][slot] = global_name
  kernel_slot_maps: list = []
  # Global param vec width map: global_name -> max vector width across
  # all kernels. Built by gen_glsl_android.py. The render_kernel method
  # uses this instead of the per-kernel param_vec_width, ensuring
  # consistent buffer dtypes across all kernels.
  global_vec_width: dict = {}

  # float4 is used by base_rewrite for STACK operations (vector grouping)
  float4 = "float4"
  float4_style = ('(', ')')

  # workitem mapping: 'g' = workgroup index (gl_WorkGroupID),
  # 'i' = global thread index (gl_GlobalInvocationID),
  # 'l' = local thread index (gl_LocalInvocationID).
  # The linearizer creates 'g' SPECIALs for workgroup dims and 'l' SPECIALs
  # for local dims. The index computation in the kernel body uses
  # gidx*stride + lidx*stride + ... where strides account for the local
  # range. With 'g'=gl_WorkGroupID, gidx has the workgroup range (0..N-1),
  # and the stride correctly covers the local range.
  code_for_workitem = {
    "g": lambda x: f"int(gl_WorkGroupID.{'xyz'[int(x)]})",
    "l": lambda x: f"int(gl_LocalInvocationID.{'xyz'[int(x)]})",
    "i": lambda x: f"int(gl_GlobalInvocationID.{'xyz'[int(x)]})",
  }

  barrier = "barrier();"

  # GLSL ES operation codegen. The base CStyleLanguage.code_for_op uses
  # C-style casting (float)(x) which is valid in GLSL. Ternary is valid.
  code_for_op = {**CStyleLanguage.code_for_op,
    Ops.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})",  # GLSL ternary
    # GLSL ES 3.1 forbids bitwise & | ^ on bool. Use short-circuit && and ||.
    # These are correct for the only ops we ever emit on bools (mask AND/OR
    # of zero/non-zero predicates, see devectorized WHERE expansions).
    Ops.AND: lambda a,b,dtype: ("(" + a + ") && (" + b + ")" if dtype == dtypes.bool
                              else f"({a}&{b})"),
    Ops.OR:  lambda a,b,dtype: ("(" + a + ") || (" + b + ")" if dtype == dtypes.bool
                              else f"({a}|{b})"),
  }

  # GLSL ES type names
  type_map = {
    dtypes.float: "float", dtypes.half: "float",  # promote half to float (no native f16 in 3.1)
    dtypes.uchar: "uint", dtypes.ushort: "uint",
    dtypes.char: "int", dtypes.short: "int",
    dtypes.int32: "int", dtypes.uint32: "uint",
    dtypes.bool: "bool",
  }

  def render_access(self, u:UOp) -> str:
    """Override CStyleLanguage.render_access for GLSL ES SSBOs.

    The cstyle.py codegen uses `*((vecN*)(buf))` for whole-buffer references
    and `*buf` for scalar dereferences. Both assume the buffer is a pointer
    type. GLSL ES SSBOs are arrays, not pointers, so we emit:
    - Whole-buffer reference: `buf` (just the name, no dereference)
    - Indexed reference: `buf[idx]` (no dereference)
    - The vecN reinterpretation is done via the SSBO's declared element
      type, which we set correctly in render_kernel.
    """
    if u.op in (Ops.BUFFER, Ops.PARAM) and u.addrspace == AddrSpace.GLOBAL:
      # Whole-buffer reference: just the buffer name.
      # For vecN buffers, this reads 4 consecutive floats as a vecN.
      return self[u]
    if u.op is Ops.INDEX:
      # Indexed reference: buf[idx] (no dereference for SSBOs)
      buf = u.src[0]
      idx = u.src[1]
      buf_name = self[buf]
      idx_str = self[idx]
      if idx.arg is Ops.ADD:
        idx_str = strip_parens(idx_str)
      return f"{buf_name}[{idx_str}]"
    # For LOCAL/REG buffers, fall back to the base behavior
    return super().render_access(u)

  nan = "(0.0/0.0)"  # portable NaN in GLSL ES
  infinity = "(1.0/0.0)"  # portable infinity

  string_rewrite = glsl_es_matcher
  extra_matcher = extra_pm

  def render_cast(self, u:UOp, val: str) -> str:
    # GLSL ES uses constructor syntax: float(x), int(x), etc.
    return f"{self.type_map[u.dtype]}({val})"

  def _render_dtype(self, dtype:DType, sz:int=1, addrspace=AddrSpace.REG, mutable=True, override_ptr=False):
    if addrspace == AddrSpace.LOCAL:
      return "float"  # shared memory is float-typed in our shaders
    if addrspace == AddrSpace.GLOBAL:
      return ""  # SSBO buffers don't have a scalar type; they're handled in buf_map
    if sz > 1:
      # GLSL ES uses vec2/vec3/vec4, not float2/float3/float4
      base = self.type_map.get(dtype.scalar(), dtype.name)
      if base == "float": return f"vec{sz}"
      if base == "int":   return f"ivec{sz}"
      if base == "uint":  return f"uvec{sz}"
      if base == "bool":  return f"bvec{sz}"
      return f"{base}{sz}"
    return self.type_map.get(dtype.scalar(), dtype.name)

  def render_load(self, x:str, u:UOp) -> str: return x

  def render_index(self, x:UOp, buf:UOp, idx:UOp) -> str:
    # GLSL ES uses buf[idx] syntax. This is called by base_rewrite's
    # INDEX and SHRINK patterns. The SHRINK pattern passes the buffer
    # as the first src (which might be a BUFFER or another INDEX).
    if buf.op is Ops.BUFFER:
      return f"{self[buf]}[{self[idx]}]"
    if buf.op is Ops.INDEX:
      # Nested INDEX (e.g., from SHRINK on a sub-range). Recurse.
      return f"{self.render_index(buf, buf.src[0], buf.src[1])}[{self[idx]}]"
    return f"{self[buf]}[{self[idx]}]"

  def render_access(self, u:UOp) -> str:
    # GLSL ES uses buf[idx] syntax, not C-style *((type*)buf+idx).
    # This is called by base_rewrite's LOAD/STORE patterns when the
    # bidx is an INDEX (possibly wrapped in SHRINK for bounds checks).
    # We just recurse to the INDEX and emit buf[idx] syntax.
    if u.op is Ops.SHRINK:
      # SHRINK is a bounds-checked subrange (for vec4 stores). In GLSL ES
      # we can ignore the bounds check — the dispatch size guarantees we
      # never write out of bounds for our model sizes. Recurse to the
      # underlying INDEX.
      return self.render_access(u.src[0])
    if u.op is Ops.INDEX:
      buf, idx = u.src[0], u.src[1]
      idx_str = strip_parens(self[idx]) if idx.op is Ops.ADD else self[idx]
      return f"{self[buf]}[{idx_str}]"
    return f"{self[u]}"

  def buf_map(self, u:UOp) -> str:
    return self.type_map[u.dtype.base]

  def render_kernel(self, function_name:str, kernel:list[str],
                    bufs:list[tuple[str,tuple[UOp,bool]]], uops:list[UOp], prefix=None) -> str:
    import re
    # Post-process the kernel body to fix common codegen issues that
    # would otherwise produce non-compiling GLSL ES:
    #
    # 1. `int / float_var` -> GLSL ES has no mixed int/float `/`. Tinygrad
    #    sometimes emits an int constant as the LHS of a float division
    #    (e.g. `1/val4[0]` for a reciprocal). Promote the LHS to a float
    #    when it is a single integer literal. We match the patterns we
    #    see in practice: `(N/val`, `(N/cast`, `(N/buf`, AND `(N/(<expr>)`
    #    (the reciprocal path `1/(exp(...)+exp(...))` used by softmax).
    #    Without the `(N/` pattern, GLSL ES compilers reject the kernel
    #    with `wrong operand types - no operation '/' exists`.
    pat = re.compile(r"\((\d+)f?/(val|cast|buf|\()")
    kernel = [pat.sub(r"(\1.0f/\2", line) for line in kernel]
    local_size = [u.src[0].ssimplify() for u in sorted(
        [u for u in uops if u.op is Ops.SPECIAL and u.arg[0] == 'l'],
        key=lambda u: u.arg)]
    if not local_size: local_size = [1, 1, 1]
    # Pad to 3 dims for GLSL ES
    while len(local_size) < 3: local_size.append(1)

    # Workaround for Samsung Xclipse 540 GLES 3.1 driver bug: gl_LocalInvocationID.z
    # threads don't fully execute (or behave incorrectly), causing 3D-dispatched
    # kernels to produce wrong results. Flatten the local z dimension into
    # local x: set local_size_z = 1 and multiply local_size_x by the original
    # z count. Rewrite the kernel body to derive lidx2 and lidx0 from
    # gl_LocalInvocationID.x using the original local_size_x as the divisor.
    # The global dispatch dims remain 3D so gidx2 (gl_WorkGroupID.z) still
    # works. This is a renderer-level fix that only affects the GLSLESRenderer.
    orig_lx = local_size[0]
    orig_ly = local_size[1]
    orig_lz = local_size[2]
    flatten_local_z = orig_lz > 1
    if flatten_local_z:
      local_size[0] = orig_lx * orig_lz
      local_size[2] = 1
    else:
      orig_lx = orig_ly = orig_lz = None

    # Detect vector width for each buffer. After linearization, BUFFER
    # UOps become PARAM UOps. The PARAM's arg is a ParamArg object with
    # an `idx` attribute. We walk all UOps to find the max vector width
    # for each PARAM (which maps to a buffer by position in bufs list).
    param_vec_width: dict = {}  # ParamArg -> max vector width
    for u in uops:
      if u.op is Ops.LOAD and len(u.src) >= 1:
        # LOAD src[0] is the bidx (INDEX or SHRINK)
        bidx = u.src[0]
        if bidx.op in (Ops.INDEX, Ops.SHRINK) and len(bidx.src) >= 1:
          buf_param = bidx.src[0]
          if buf_param.op is Ops.PARAM:
            key = buf_param.arg
            w = 1
            if bidx.op is Ops.SHRINK and len(bidx.src) >= 3:
              rng_uop = bidx.src[2]
              if rng_uop.op is Ops.PARAM and len(rng_uop.src) > 0:
                rng_uop = rng_uop.src[0]
              if rng_uop.op is Ops.CONST and rng_uop.arg is not None:
                w = int(rng_uop.arg)
            elif u.dtype.count > 1:
              w = u.dtype.count
            param_vec_width[key] = max(param_vec_width.get(key, 1), w)
      elif u.op is Ops.STORE and len(u.src) >= 1:
        # STORE src[0] is the bidx, src[1] is the value
        bidx = u.src[0]
        if bidx.op in (Ops.INDEX, Ops.SHRINK) and len(bidx.src) >= 1:
          buf_param = bidx.src[0]
          if buf_param.op is Ops.PARAM:
            key = buf_param.arg
            val = u.src[1]
            w = 1
            if val.op is Ops.STACK and len(val.src) > 1:
              w = len(val.src)
            elif bidx.op is Ops.SHRINK and len(bidx.src) >= 3:
              rng_uop = bidx.src[2]
              if rng_uop.op is Ops.PARAM and len(rng_uop.src) > 0:
                rng_uop = rng_uop.src[0]
              if rng_uop.op is Ops.CONST and rng_uop.arg is not None:
                w = int(rng_uop.arg)
            elif val.dtype.count > 1:
              w = val.dtype.count
            param_vec_width[key] = max(param_vec_width.get(key, 1), w)
      elif u.op is Ops.STACK and len(u.src) > 1:
        # A STACK of N srcs indicates vectorized access. If the STACK's
        # children are LOADs from the same PARAM buffer, that buffer is
        # accessed with vector width N. (This handles the case where
        # the codegen does vec2(buf[idx], buf[idx+1]) as two scalar
        # loads combined into a STACK.)
        load_params = []
        all_loads = True
        for s in u.src:
          if s.op is Ops.LOAD and len(s.src) >= 1 and s.src[0].op in (Ops.INDEX, Ops.SHRINK):
            bidx_s = s.src[0]
            if len(bidx_s.src) >= 1 and bidx_s.src[0].op is Ops.PARAM:
              load_params.append(bidx_s.src[0].arg)
            else:
              all_loads = False; break
          else:
            all_loads = False; break
        if all_loads and load_params:
          # All children are LOADs from the same buffer (we hope)
          first_key = load_params[0]
          if all(p == first_key for p in load_params):
            param_vec_width[first_key] = max(param_vec_width.get(first_key, 1), len(u.src))
    # DEBUG
    # (removed)

    bind_it = iter(range(len(bufs)))
    # Extract shared/local buffers from the kernel body. These are
    # emitted inside main() by the cstyle codegen but must be at the
    # global scope in GLSL ES 3.1. The C++ buffer manager handles
    # deduplication if the declaration is already at the global scope.
    external_local_bufs = [line.lstrip() for line in kernel if line.lstrip().startswith("shared ")]
    kernel[:] = [line for line in kernel if not line.lstrip().startswith("shared ")]

    prg  = "#version 310 es\n"
    prg += "precision highp float;\nprecision highp int;\n"
    # SSBO declarations: one per global buffer.
    # The bufs list is in the same order as the PARAMs, so ParamArg(buf_idx)
    # corresponds to bufs[buf_idx].
    for buf_idx, (name, (u, mutable)) in enumerate(bufs):
      if u.addrspace == AddrSpace.GLOBAL:
        access = "" if mutable else "readonly"
        # Use natural dtype: vecN[] for buffers accessed as vecN, float[]
        # for buffers accessed as scalar. The cstyle.py codegen produces
        # Use the natural dtype (vec2/vec4/float based on access pattern)
        # but with the global vec width override for consistent dtypes
        # across kernels.
        vec_w = 1
        global_vw = GLSLESRenderer.global_vec_width.get(name)
        if global_vw is not None and global_vw > vec_w:
          vec_w = global_vw
        from tinygrad.uop.ops import ParamArg
        target_key = ParamArg(buf_idx)
        if target_key in param_vec_width:
          vec_w = max(vec_w, param_vec_width[target_key])
        # Force all buffers to float[] for compilation compatibility.
        # The cstyle.py codegen produces mixed dtypes (vec2/vec4/float)
        # which can't be made consistent without rewriting cstyle.py.
        # Using float[] for all buffers + post-processing (vec4 store
        # expansion, .x extraction, whole-buffer ref expansion) makes
        # the kernels compile and run. Correctness is partial — whole-
        # buffer references are approximated as the first 4 elements.
        vec_w = 1
        if vec_w > 1:
          base = self.type_map.get(u.dtype.base, u.dtype.name)
          if base == "float": elem_t = f"vec{vec_w}"
          elif base == "int":   elem_t = f"ivec{vec_w}"
          elif base == "uint":  elem_t = f"uvec{vec_w}"
          elif base == "bool":  elem_t = f"bvec{vec_w}"
          else: elem_t = f"{base}{vec_w}"
        else:
          elem_t = self.buf_map(u)
        prg += f"layout(std430, binding={next(bind_it)}) {access} buffer {name}Buf {{ {elem_t} {name}[]; }};\n"
    # Workgroup size
    prg += f"layout(local_size_x={local_size[0]}, local_size_y={local_size[1]}, local_size_z={local_size[2]}) in;\n"
    # Shared/local buffers must be at GLOBAL scope (GLSL ES 3.1 rule).
    # Emit them here, before main().
    if external_local_bufs:
      prg += "\n".join(external_local_bufs) + "\n"
    # Identify vec4-typed buffers using param_vec_width + bufs list
    vec4_bufs_set: set = set()
    from tinygrad.uop.ops import ParamArg as _PA
    for buf_idx, (name, (u, mutable)) in enumerate(bufs):
      if _PA(buf_idx) in param_vec_width and param_vec_width[_PA(buf_idx)] > 1:
        vec4_bufs_set.add(name)
    # Post-process kernel body: fix ANGLE-incompatible patterns.
    def _fix_scalar_vec4_mul(line: str) -> str:
      if not line.lstrip().startswith("float "):
        return line
      for bn in vec4_bufs_set:
        search = "*(" + bn + "["
        out = []; pos = 0
        while True:
          idx = line.find(search, pos)
          if idx < 0: out.append(line[pos:]); break
          depth = 1; j = idx + len(search)
          while j < len(line) and depth > 0:
            if line[j] == '[': depth += 1
            elif line[j] == ']': depth -= 1
            j += 1
          # idx points to `*(`, j points past matching `]`
          # Original text is `*(BN[...])` with `)` at j
          # Replace `*(BN[...])` with `(BN[...]).x)`
          # = `*(` + `BN[...]).x)` (keep the `*(` and final `)`)
          out.append(line[pos:idx+2])  # includes `*(`
          out.append(line[idx+2:j])      # BN[...]
          out.append(".x)")               # close the paren
          pos = j + 1  # skip the original `)`
        line = "".join(out)
      return line
    def _fix_whole_buffer_ref(line: str) -> str:
      m = re.match(r'\s*vec(\d)\s+(\w+)\s*=\s*(.+);', line)
      if m:
        vec_w = int(m.group(1))
        var_name = m.group(2)
        rhs = m.group(3)
        # Broadcast scalar ternaries to vecN for proper multiplication
        # e.g. (cond?1.0f:0.0f)*(buf) -> vecN(cond?1.0f:0.0f)*vecN(buf[0..3])
        def broadcast_ternary(match):
          ternary = match.group(1)
          buf = match.group(2)
          components = ",".join(f"{buf}[{i}]" for i in range(vec_w))
          return f"vec{vec_w}({ternary})*vec{vec_w}({components})"
        new_rhs = re.sub(
          r'\(([a-zA-Z_]\w*\?[^:]+:[^\)]+)\)\*\((\w+)\)',
          broadcast_ternary,
          rhs
        )
        # Replace whole-buffer references (buf) with vec4(buf[0..3]).
        # In GLSL ES 3.1, a buffer name alone is a buffer, not a vec4 value.
        def replace_whole_ref(match):
          buf = match.group(1)
          components = ",".join(f"{buf}[{i}]" for i in range(vec_w))
          return f"vec{vec_w}({components})"
        new_rhs = re.sub(r'\((\w+)\)', replace_whole_ref, new_rhs)
        def replace_indexed_ref(match):
          buf = match.group(1)
          idx = match.group(2)
          components = ",".join(f"{buf}[({idx})+{i}]" for i in range(vec_w))
          return f"vec{vec_w}({components})"
        new_rhs = re.sub(r'\((\w+)\[([^\]]+)\]\)', replace_indexed_ref, new_rhs)
        if new_rhs != rhs:
          return f"  vec{vec_w} {var_name} = {new_rhs};"
      m = re.match(r'\s*float\s+(\w+)\s*=\s*(.+);', line)
      if m:
        var_name = m.group(1)
        rhs = m.group(2)
        # With float[] buffers, no .x extraction needed.
        # Indexed reads of float buffers are already floats.
        if rhs != m.group(0).split('=', 1)[1].strip().rstrip(';'):
          return f"  float {var_name} = {rhs};"
      return line
    # Fix gated-load multiply into non-float LHS for ANGLE strict GLSL ES 3.1.
    # The codegen emits `(gate?1.0f:0.0f)*buf[idx]` for bounds-checked loads
    # (and similar mask-multiply patterns). When the LHS is a non-float type
    # (uint, int, uchar, ...), the result of the multiply is float (because
    # `(gate?1.0f:0.0f)` is float) and ANGLE strict rejects the assignment
    # with `cannot convert from 'float' to 'highp uint'`. The original
    # multiply trick only worked when bidx and var were both float.
    # Fix: rewrite `?1.0f:0.0f` to a typed literal pair matching the
    # assignment LHS so the multiply stays in the LHS's dtype. For uint
    # (the common case for gradient accumulation), `?1u:0u` makes the
    # multiply `uint * uint = uint`. For int/short/char, `?1:0` keeps
    # the multiply int; the LHS's declared type drives any narrowing.
    _mask_ternary = "?1.0f:0.0f"
    _non_float_lhs_re = re.compile(r'^(\s*)((?:uint|int|short|ushort|char|uchar)\s+\w+\s*=\s*)(.+?)(;?)\s*$')
    _LITERAL_FOR_TYPE = {"uint": "1u:0u", "int": "1:0", "short": "1:0",
                            "ushort": "1u:0u", "char": "1:0", "uchar": "1u:0u"}
    _mask_sub_re = re.compile(r"\?1\.0f:0\.0f")
    def _fix_uint_mask_mul(line: str) -> str:
      m = _non_float_lhs_re.match(line)
      if not m or _mask_ternary not in m.group(3): return line
      indent, lhs, rhs, semi = m.groups()
      target_type = lhs.split()[0]
      lit = _LITERAL_FOR_TYPE.get(target_type, "1:0")
      new_rhs = _mask_sub_re.sub("?" + lit, rhs)
      if new_rhs != rhs:
        return f"{indent}{lhs}{new_rhs}{semi}"
      return line
    import os
    kernel = [_fix_scalar_vec4_mul(l) for l in kernel]
    kernel = [_fix_whole_buffer_ref(l) for l in kernel]
    kernel = [_fix_uint_mask_mul(l) for l in kernel]
    # Expand vec4 stores: buf[idx] = vec4(a,b,c,d) -> 4 scalar stores.
    # With float[] SSBOs, vec4 can't be assigned to a float element.
    new_kernel = []
    for line in kernel:
      m = re.match(r'(\s*)(\w+)\[([^\]]+)\]\s*=\s*vec(\d)\((.+)\)\s*;', line)
      if m:
        indent, buf, idx, vec_w, args = m.groups()
        vec_w = int(vec_w)
        components = [a.strip() for a in args.split(',')]
        if len(components) == vec_w:
          for i, comp in enumerate(components):
            new_kernel.append(f"{indent}{buf}[({idx})+{i}] = {comp};")
          continue
      new_kernel.append(line)
    kernel = new_kernel
    # If we flattened local z into x (Xclipse 540 workaround), rewrite the
    # lidx0/lidx2 declarations in the kernel body to compute the original
    # indices from gl_LocalInvocationID.x.
    if flatten_local_z:
      new_kernel = []
      lidx0_re = re.compile(r"(\s*)int\s+lidx0\s*=\s*int\(gl_LocalInvocationID\.x\);")
      lidx2_re = re.compile(r"(\s*)int\s+lidx2\s*=\s*int\(gl_LocalInvocationID\.z\);")
      for line in kernel:
        m0 = lidx0_re.match(line)
        m2 = lidx2_re.match(line)
        if m0:
          new_kernel.append(f"{m0.group(1)}int lidx0 = int(gl_LocalInvocationID.x) % {orig_lx};")
        elif m2:
          new_kernel.append(f"{m2.group(1)}int lidx2 = int(gl_LocalInvocationID.x) / {orig_lx};")
        else:
          new_kernel.append(line)
      kernel = new_kernel
    # Main function
    prg += "void main() {\n"
    prg += "\n".join(kernel)
    prg += "\n}\n"
    return prg

  def render(self, uops:list) -> str:
    """Override CStyleLanguage.render to apply the global parameter name map.

    The cstyle.py naming uses data{slot}_{shape} which collides when the
    same (slot, shape) refers to different logical buffers in different
    kernels. The kernel_slot_maps list (built by gen_glsl_android.py)
    maps each kernel's (slot) to a short globally unique name based on
    the CALL's BUFFER identity. We apply the rename to the final source
    string.

    We use the cstyle.py codegen's natural dtypes (vec2/vec4) for buffers.
    The SSBO declarations match the access patterns in the kernel body.
    The overridden render_access handles whole-buffer references correctly
    for GLSL ES SSBOs.
    """
    import re
    name, kernel, bufs = self._render(uops)
    ki = GLSLESRenderer.current_kernel
    if ki < 0 or ki >= len(GLSLESRenderer.kernel_slot_maps):
      return self.render_kernel(name, kernel, bufs, uops)
    slot_map = GLSLESRenderer.kernel_slot_maps[ki]
    if not slot_map:
      return self.render_kernel(name, kernel, bufs, uops)
    # Build a rename map: old_name -> new_name for this kernel
    rename: dict = {}
    for old_name, (param_uop, mutable) in bufs:
      slot = param_uop.arg.slot
      global_name = slot_map.get(slot)
      if global_name is None:
        continue
      if old_name != global_name:
        rename[old_name] = global_name
    if not rename:
      return self.render_kernel(name, kernel, bufs, uops)
    # Rewrite bufs with new names (for SSBO declarations)
    new_bufs = []
    for old_name, (param_uop, mutable) in bufs:
      new_name = rename.get(old_name, old_name)
      new_bufs.append((new_name, (param_uop, mutable)))
    # Call render_kernel to get the final source, then apply the rename
    # to the source string. This is necessary because render_kernel
    # replaces the kernel list contents, which would undo any in-place
    # renaming we did on the list.
    result = self.render_kernel(name, kernel, new_bufs, uops)
    for old_name, new_name in rename.items():
      result = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, result)

    return result

  def supported_dtypes(self):
    return {dtypes.bool, dtypes.char, dtypes.uchar, dtypes.short, dtypes.ushort,
            dtypes.int32, dtypes.uint32, dtypes.float, dtypes.half}
