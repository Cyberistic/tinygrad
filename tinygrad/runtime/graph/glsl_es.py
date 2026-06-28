from typing import cast
from tinygrad.helpers import PROFILE, perf_counter_us, cpu_events
from tinygrad.device import Device, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.uop.ops import UOp, Ops
from tinygrad.engine.jit import GraphRunner
from tinygrad.runtime.ops_glsl_es import GLSL_ESDevice, GLSLESProgram, _FFI

class GLSLESGraph(GraphRunner):
  def __init__(self, linear:UOp, input_uops:tuple[UOp, ...]=()):
    super().__init__(linear, input_uops)
    self.dev = cast(GLSL_ESDevice, Device[self.device])

  def __call__(self, input_uops:tuple[UOp, ...], var_vals:dict[str, int], wait=False) -> float|None:
    st = perf_counter_us()
    for (_, ast, bufs, device_vars), runtime in zip(self.calls, self.runtimes):
      if ast.op is not Ops.PROGRAM: continue
      prg_bufs = [bufs[i].ensure_allocated() for i in ast.arg.globals]
      rt = cast(GLSLESProgram, runtime)
      global_size, local_size = ast.arg.launch_dims({**var_vals, **device_vars})
      rt(*[b.get_buf(self.device) for b in prg_bufs], global_size=global_size, local_size=local_size,
         vals=ast.arg.vals({**var_vals, **device_vars}), wait=False)
    if _FFI.is_native: self.dev.synchronize()
    if not wait and not PROFILE: return None
    en = perf_counter_us()
    if wait and _FFI.is_native:
      return max(1e-6, (en - st) * 1e-6)
    if PROFILE:
      n = sum(1 for c in self.calls if c[1].op is Ops.PROGRAM)
      dur = max(1, (en - st) // max(n, 1))
      sigs = [st + i * dur for i in range(n + 1)]
      ents = [ProfileGraphEntry(self.device, cast(GLSLESProgram, rt).name, 2 * i, 2 * i + 1)
              for i, ((_, ast, _, _), rt) in enumerate(zip(self.calls, self.runtimes)) if ast.op is Ops.PROGRAM]
      cpu_events.append(ProfileGraphEvent(ents, [], sigs))
    return None