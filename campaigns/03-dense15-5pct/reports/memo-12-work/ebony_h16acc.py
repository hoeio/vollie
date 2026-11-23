# half-precision accumulation of a K-term Q4_0 dot product, as gemm_noshuffle_q4_0_f32.cl does (half8 c += B * w),
# against float64 on the same half-rounded inputs. Random data only. Unfused (product rounded to half, then add).
import numpy as np
rng = np.random.default_rng(0)
T = 4000
for K, heavy in ((4096, 0), (11008, 0), (4096, 1), (11008, 1)):
    x = rng.standard_normal((T, K))
    if heavy: x *= np.exp(rng.standard_normal((T, K)))           # log-normal spread of magnitudes
    x = x.astype(np.float16)
    q = rng.integers(0, 16, (T, K)) - 8
    d = np.repeat((np.abs(rng.standard_normal((T, K // 32))) * 0.02 + 0.005), 32, axis=1).astype(np.float16)
    w = (q.astype(np.float16) * d).astype(np.float16)             # (q - 8) * d in half
    ref = (x.astype(np.float64) * (q * d.astype(np.float64))).sum(1)
    acc = np.zeros(T, np.float16)
    for i in range(K):
        acc = (acc + (x[:, i] * w[:, i]).astype(np.float16)).astype(np.float16)
    err = acc.astype(np.float64) - ref
    print(f"K={K} heavy={heavy} rms(err)/rms(ref)={np.sqrt((err**2).mean())/np.sqrt((ref**2).mean()):.4f} "
          f"p99|err|/rms(ref)={np.quantile(np.abs(err),0.99)/np.sqrt((ref**2).mean()):.4f} nonfinite={(~np.isfinite(acc)).sum()}")
# self-check: float32 accumulation of the same must be ~1e-6
x = rng.standard_normal((50, 4096)).astype(np.float16); w = rng.standard_normal((50, 4096)).astype(np.float16)
a = np.zeros(50, np.float32)
for i in range(4096): a = a + x[:, i].astype(np.float32) * w[:, i].astype(np.float32)
r = (x.astype(np.float64) * w.astype(np.float64)).sum(1)
assert np.sqrt(((a - r) ** 2).mean()) / np.sqrt((r ** 2).mean()) < 1e-5
print("selfcheck ok")
