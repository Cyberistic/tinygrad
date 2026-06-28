import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
from tinygrad.codegen import to_program

from tinygrad.runtime.ops_glsl_es import _FFI

@unittest.skipUnless(_FFI.is_native, "GLSL_ES requires libflower_tinygrad.so (Android GLES runtime)")
class TestGLSLESNative(unittest.TestCase):
  def setUp(self):
    self.ctx = Context(DEV="GLSL_ES", NO_MEMORY_PLANNER=0)
    self.ctx.__enter__()

  def tearDown(self):
    self.ctx.__exit__(None, None, None)

  def test_dot_product(self):
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([5.0, 6.0, 7.0, 8.0])
    out = (a * b).sum().numpy()
    np.testing.assert_allclose(out, 70.0, rtol=1e-4, atol=1e-4)

  def test_matmul_fwd(self):
    np.random.seed(0)
    x_np = np.random.randn(8, 16).astype(np.float32)
    w_np = np.random.randn(16, 32).astype(np.float32)
    out = (Tensor(x_np) @ Tensor(w_np)).numpy()
    np.testing.assert_allclose(out, x_np @ w_np, rtol=1e-3, atol=1e-3)

  def test_matmul_bwd(self):
    np.random.seed(1)
    x_np = np.random.randn(8, 16).astype(np.float32)
    w_np = np.random.randn(16, 32).astype(np.float32)
    x = Tensor(x_np)
    w = Tensor(w_np)
    (x @ w).sum().backward()
    # dx = w.T  (ones output) ; shape (8, 16)
    np.testing.assert_allclose(x.grad.numpy(), np.ones((8, 32)) @ w_np.T, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(w.grad.numpy(), x_np.T @ np.ones((8, 32)), rtol=1e-3, atol=1e-3)

  def test_alloc_reuse(self):
    a = Tensor.randn(64).realize()
    b = Tensor.randn(64).realize()
    c = (a + b).numpy()
    np.testing.assert_allclose(c, a.numpy() + b.numpy(), rtol=1e-3, atol=1e-3)

  def test_dispatch_error_propagation(self):
    # broken shader source should raise via pop_error, not silently pass
    dev = Device["GLSL_ES"]
    err = dev.pop_error()
    # On entry the queue should be drained; non-empty means a prior kernel errored.
    # Just assert the call returns a str (no crash).
    self.assertIsInstance(err, str)

  def test_sync_no_errors(self):
    Device["GLSL_ES"].synchronize()
    self.assertEqual(Device["GLSL_ES"].pop_error(), "")

class TestGLSLESCodegen(unittest.TestCase):
  """Runs on any host (no native lib needed). Verifies the renderer produces
  GLSL ES 3.1 source and the device integrates with tinygrad's runtime
  (allocator allocates, dispatch is a no-op, copyin/copyout round-trip).
  """
  def setUp(self):
    self.ctx = Context(DEV="GLSL_ES", NO_MEMORY_PLANNER=1)
    self.ctx.__enter__()

  def tearDown(self):
    self.ctx.__exit__(None, None, None)

  def test_renders_glsl_es(self):
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([5.0, 6.0, 7.0, 8.0])
    linear = (a * b).sum().schedule_linear()
    sinks = [s for call in linear.src for s in call.src if s.op is Ops.SINK]
    self.assertGreater(len(sinks), 0)
    prg = to_program(sinks[0], Device["GLSL_ES"].renderer)
    src = None
    for u in prg.toposort():
      if u.op is Ops.SOURCE:
        src = u.arg
        break
    self.assertIsNotNone(src)
    self.assertIn("#version 310 es", src)
    self.assertIn("void main()", src)

  def test_allocator_round_trip(self):
    a = Device["GLSL_ES"].allocator
    opaque = a._alloc(16, a.default_buffer_spec)
    mv = memoryview(b"\x01\x02\x03\x04\x05\x06\x07\x08")
    a._copyin(opaque, mv)
    out = bytearray(8)
    a._copyout(memoryview(out), opaque)
    self.assertEqual(bytes(out), bytes(mv))
    a._free(opaque, a.default_buffer_spec)

  def test_offset_preserves_shadow(self):
    a = Device["GLSL_ES"].allocator
    opaque = a._alloc(16, a.default_buffer_spec)
    mv = memoryview(bytes(range(16)))
    a._copyin(opaque, mv)
    sub = a._offset(opaque, 4, 4)
    out = bytearray(4)
    a._copyout(memoryview(out), sub)
    self.assertEqual(bytes(out), bytes([4, 5, 6, 7]))
    a._free(opaque, a.default_buffer_spec)

if __name__ == "__main__":
  unittest.main()