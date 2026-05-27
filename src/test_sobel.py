import numpy as np
from scipy.ndimage import sobel
from time import time

# --------------------------------------------------------
# 1. Создаём большой тестовый массив 20000 × 20000
# --------------------------------------------------------
print("Creating array...")
t0 = time()
A = np.random.rand(30000, 30000).astype(np.float32)
print(f"Created in {time() - t0:.2f}s | size = {A.nbytes/1e9:.2f} GB")

# ========================================================
# TEST 1 — Sobel по axis=0 (C-order)
# ========================================================
print("\n=== Test 1: C-order sobel(A, axis=0) ===")
t1 = time()
gy_c = sobel(A, axis=0)
print(f"Sobel C-order: {time() - t1:.3f}s")

# освободим память перед F-order
del gy_c
import gc; gc.collect()

# ========================================================
# TEST 2 — Sobel по axis=0 (F-order)
# ========================================================
print("\n=== Test 2: F-order sobel(A_F, axis=0) ===")
t2 = time()
A_F = np.asfortranarray(A)      # Перекладка в Fortran-order (создаёт копию)
print(f"Converted to F-order in {time() - t2:.3f}s")

t3 = time()
gy_f = sobel(A_F, axis=0)
print(f"Sobel F-order: {time() - t3:.3f}s")

# итог
print("\n=== RESULT SUMMARY ===")
print(f"C-order Sobel:   {time() - t1:.3f}s (см. выше)")
print(f"F-order Sobel:   {time() - t3:.3f}s")

