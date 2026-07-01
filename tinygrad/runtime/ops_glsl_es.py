"""GLSL ES 3.1 device for tinygrad (Android GLES runtime).

Emits GLSL ES 3.1 compute shaders via `GLSLESRenderer`. Execution is driven
through `libflower_tinygrad.so` (the JNI shim in `app/src/main/cpp/`), which
owns the EGL/GLES 3.1 context and the SSBO buffer pool. When the native lib
is not loadable (e.g. a dev host without the .so), the device degrades to a
codegen-only path: rendering still runs, allocation/dispatch are no-ops, and
`copyout` returns the zero-initialised host shadow. This lets the rest of the
runtime (linearizer, JIT, tests) run end-to-end on a laptop.

Native API (see `app/src/main/cpp/native_api.h` + `gpu_ops_jit.h`):
  android_gles_init / shutdown / is_available / renderer
  android_gles_buffer_create / free / upload / download
  dispatch_jit_kernel(src, buf_handles, bindings, n_bufs, gx, gy, gz)
"""
from __future__ import annotations
import ctypes, functools, os

import numpy as np

from tinygrad.device import Compiled, LRUAllocator, BufferSpec
from tinygrad.renderer.glsl_es import GLSLESRenderer

_LIB_NAMES = ("libflower_tinygrad.so", "flower_tinygrad.so")

def _mv_as_ptr(mv: memoryview) -> ctypes.c_void_p:
  return ctypes.c_void_p(np.frombuffer(mv, dtype=np.uint8).ctypes.data)

def _np_as_ptr(arr: np.ndarray) -> ctypes.c_void_p:
  return ctypes.c_void_p(arr.ctypes.data)

class _NativeFFI:
  """Lazy ctypes binding to `libflower_tinygrad.so`. On any failure to load
  or call, the device falls back to codegen-only mode and every method
  returns the safe no-op default shown in `__init__`."""
  def __init__(self) -> None:
    self.lib: ctypes.CDLL | None = None
    self.is_native: bool = False
    self.renderer_str: str = ""
    self._bind()

  def _ensure_bound(self) -> None:
    """Retry the native lib load if it failed at import time. This handles
    Chaquopy use cases where the lib is staged to the app files dir AFTER
    the Python runtime has already imported tinygrad (so FLOWER_TINYGRAD_LIB
    was not yet set when __init__ ran). The first FFI call re-attempts the
    load with the current environment."""
    if self.lib is not None and self.is_native: return
    self.lib = None
    self.is_native = False
    self._bind()

  def _bind(self) -> None:
    candidates: list[str] = []
    if (p := os.environ.get("ANDROID_GLES_NATIVE_LIB") or os.environ.get("FLOWER_TINYGRAD_LIB")): candidates.append(p)
    candidates += list(_LIB_NAMES)
    for path in candidates:
      try: self.lib = ctypes.CDLL(path)
      except OSError: continue
      if self._configure(): break
      self.lib = None
    if self.lib is None: return
    try:
      # init returns 1 on success, 0 on failure (matches gpu_ops::init()).
      if self.lib.android_gles_init() != 0:
        self.is_native = bool(self.lib.android_gles_is_available())
        if self.is_native:
          self.renderer_str = ctypes.string_at(self.lib.android_gles_renderer()).decode()
    except Exception: self.is_native = False

  def _configure(self) -> bool:
    # Set argtypes/restypes for each symbol that exists. Symbols that are
    # missing (older libflower_tinygrad.so versions that lack the newer
    # android_gles_last_kernel_time_ns / android_gles_pop_error exports)
    # are skipped so the device still works on older libs.
    lib = self.lib
    for name, restype, argtypes in [
      ("android_gles_init", ctypes.c_int, []),
      ("android_gles_shutdown", None, []),
      ("android_gles_is_available", ctypes.c_int, []),
      ("android_gles_renderer", ctypes.c_char_p, []),
      ("android_gles_buffer_create", ctypes.c_uint32, [ctypes.c_size_t]),
      ("android_gles_buffer_free", None, [ctypes.c_uint32]),
      ("android_gles_buffer_upload", ctypes.c_int, [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t]),
      ("android_gles_buffer_download", ctypes.c_int, [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t]),
      ("dispatch_jit_kernel", ctypes.c_bool, [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32),
                                              ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                              ctypes.c_int, ctypes.c_int, ctypes.c_int]),
      ("android_gles_last_kernel_time_ns", ctypes.c_int64, []),
      ("android_gles_pop_error", ctypes.c_char_p, []),
    ]:
      try:
        fn = getattr(lib, name)
      except AttributeError:
        continue
      fn.restype = restype
      fn.argtypes = argtypes
    return True

  def shutdown(self) -> None:
    if self.lib is None: return
    try: self.lib.android_gles_shutdown()
    except Exception: pass
    self.is_native, self.renderer_str = False, ""

  def create_buffer(self, bytes_n: int) -> int:
    self._ensure_bound()
    if not self.is_native: return 0
    return int(self.lib.android_gles_buffer_create(ctypes.c_size_t(bytes_n)))

  def free_buffer(self, handle: int) -> None:
    if self.lib is None or handle == 0: return
    try: self.lib.android_gles_buffer_free(ctypes.c_uint32(handle))
    except Exception: pass

  def upload(self, handle: int, host: memoryview | np.ndarray) -> int:
    self._ensure_bound()
    if not self.is_native or handle == 0: return 0
    if isinstance(host, np.ndarray): ptr = _np_as_ptr(host)
    else: ptr = _mv_as_ptr(host)
    return int(self.lib.android_gles_buffer_upload(ctypes.c_uint32(handle), ptr, ctypes.c_size_t(len(host))))

  def download(self, handle: int, host: np.ndarray) -> int:
    self._ensure_bound()
    if not self.is_native or handle == 0:
      host[:] = 0
      return 0
    return int(self.lib.android_gles_buffer_download(ctypes.c_uint32(handle), _np_as_ptr(host), ctypes.c_size_t(host.nbytes)))

  def dispatch_jit(self, src: str, buf_handles: list[int], gx: int, gy: int, gz: int) -> bool:
    self._ensure_bound()
    if not self.is_native: return False
    n = len(buf_handles)
    handles = (ctypes.c_uint32 * n)(*buf_handles) if n else None
    bindings = (ctypes.c_int * n)(*(range(n))) if n else None
    return bool(self.lib.dispatch_jit_kernel(src.encode(), handles, bindings, n, gx, gy, gz))
  def last_kernel_time_ns(self) -> int:
    self._ensure_bound()
    if not self.is_native: return 0
    return int(self.lib.android_gles_last_kernel_time_ns())
  def pop_error(self) -> str:
    self._ensure_bound()
    if not self.is_native: return ""
    try: return bytes(self.lib.android_gles_pop_error()).decode()
    except Exception: return ""

_FFI = _NativeFFI()

class GLSLESAllocator(LRUAllocator['GLSL_ESDevice']):
  """Allocator backed by GLES SSBOs. Opaque buffer is a
  `(handle:int, shadow:np.ndarray[uint8])` tuple. `shadow` is a CPU mirror
  used for zero-copy `as_memoryview` and as the host target of downloads."""
  def _alloc(self, size:int, options:BufferSpec) -> tuple[int, np.ndarray]:
    handle = _FFI.create_buffer(size)
    shadow = np.empty(size, dtype=np.uint8)
    shadow.fill(0)
    if handle == 0 and _FFI.is_native: raise MemoryError(f"GLES OOM while allocating {size=} bytes")
    return (handle, shadow)
  def _free(self, opaque:tuple[int, np.ndarray], options:BufferSpec) -> None:
    handle, _ = opaque
    _FFI.free_buffer(handle)
  def _offset(self, buf:tuple[int, np.ndarray], size:int, offset:int) -> tuple[int, np.ndarray]:
    handle, shadow = buf
    return (handle, shadow[offset:offset+size])
  def _as_buffer(self, src:tuple[int, np.ndarray]) -> memoryview:
    return memoryview(src[1])
  def _copyin(self, dest:tuple[int, np.ndarray], src:memoryview) -> None:
    handle, shadow = dest
    shadow[:src.nbytes] = np.frombuffer(src, dtype=np.uint8)
    _FFI.upload(handle, shadow[:src.nbytes])
  def _copyout(self, dest:memoryview, src:tuple[int, np.ndarray]) -> None:
    handle, shadow = src
    if _FFI.is_native and handle != 0: _FFI.download(handle, shadow)
    dest[:] = shadow[:dest.nbytes]
  def _transfer(self, dest:tuple[int, np.ndarray], src:tuple[int, np.ndarray], sz:int,
                src_dev:'GLSL_ESDevice', dest_dev:'GLSL_ESDevice') -> None:
    src_handle, src_shadow = src
    if _FFI.is_native and src_handle != 0:
      from tinygrad.helpers import cpu_profile
      with cpu_profile(f"{src_dev.device} -> {dest_dev.device}", f"{src_dev.device}:SDMA:0"):
        _FFI.download(src_handle, src_shadow)
        dest_dev.synchronize()
    else:
      src_dev.synchronize()
    _, dest_shadow = dest
    dest_shadow[:sz] = src_shadow[:sz]
    if _FFI.is_native:
      _FFI.upload(dest[0], dest_shadow[:sz])
  def _map(self, buf:tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
    return buf
  def _unmap(self, mapped_buf:tuple[int, np.ndarray]) -> None:
    return

class GLSLESProgram:
  def __init__(self, dev:'GLSL_ESDevice', name:str, lib:bytes, **kwargs) -> None:
    self.dev, self.name = dev, name
    self.src = lib.decode() if isinstance(lib, bytes) else lib
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1),
               local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw) -> float|None:
    if not _FFI.is_native:
      if wait: return 0.0
      return None
    handles = [b[0] if isinstance(b, tuple) else b for b in bufs]
    gx, gy, gz = global_size
    if not _FFI.dispatch_jit(self.src, handles, gx, gy, gz):
      err = _FFI.pop_error()
      # Dump the failing source for debugging
      try:
        with open("/data/user/0/flwr.tinygrad_client/files/failing_shader.glsl", "w") as f:
          f.write(f"// {self.name} {global_size}\n// Error: {err}\n\n")
          f.write(self.src)
      except Exception: pass
      raise RuntimeError(f"JIT dispatch failed for {self.name}: {err}")
    if wait: return _FFI.last_kernel_time_ns() * 1e-9
    return None

class GLSL_ESDevice(Compiled):
  def __init__(self, device:str) -> None:
    self._renderer_string = _FFI.renderer_str
    from tinygrad.runtime.graph.glsl_es import GLSLESGraph
    super().__init__(device, GLSLESAllocator(self), [GLSLESRenderer],
                     functools.partial(GLSLESProgram, self), graph=GLSLESGraph, arch="")
  @property
  def renderer_string(self) -> str:
    _FFI._ensure_bound()
    return _FFI.renderer_str
  @property
  def native(self) -> bool:
    # Probe FFI lazily so the value reflects rebinds that happened after
    # this device was constructed (e.g. env var set after tinygrad import).
    _FFI._ensure_bound()
    return _FFI.is_native
  def synchronize(self) -> None:
    _FFI._ensure_bound()
    if not _FFI.is_native: return
    if (err := _FFI.pop_error()): raise RuntimeError(f"GLES error: {err}")
  def finalize(self) -> None: _FFI.shutdown()
  def supports_mem_planner(self) -> bool: return False
  def pop_error(self) -> str:
    _FFI._ensure_bound()
    return _FFI.pop_error()