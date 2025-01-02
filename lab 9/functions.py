import numpy as np
from PIL import Image
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def gradients(image_path):
    image_np = np.array(Image.open(image_path).convert('L'), dtype=np.float32)
    gx = np.pad(np.diff(image_np, axis=1), ((0, 0), (0, 1)), mode='constant')
    gy = np.pad(np.diff(image_np, axis=0), ((0, 1), (0, 0)), mode='constant')
    return image_np, gx, gy

def poisson_reconstruction(grad_x, grad_y, iterations=200, tol=1e-3):
    h, w = grad_x.shape
    f = np.zeros((h, w)) 
    for it in range(iterations):
        f_prev = f.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                f[y, x] = 0.25 * (f[y, x + 1] + f[y, x - 1] + f[y + 1, x] + f[y - 1, x]
                                  - (grad_x[y, x] - grad_x[y, x - 1])
                                  - (grad_y[y, x] - grad_y[y - 1, x]))
        
        diff = np.linalg.norm(f - f_prev) / (np.linalg.norm(f_prev) if np.linalg.norm(f_prev) != 0 else 1)
        if diff < tol:
            print(f'Accuracy reached. {it} iter, diff {diff:.6f}.')
            break
    return f

def reconstruct_image_sparse(grad_x, grad_y):
    h, w = grad_x.shape
    size = h * w
    A = lil_matrix((size, size), dtype=np.float32)
    b = np.zeros(size, dtype=np.float32)
    idx = lambda y, x: y * w + x

    for y in range(h):
        for x in range(w):
            i = idx(y, x)
            if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                A[i, i] = 1
            else:
                A[i, i] = 4
                for ny, nx in [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]:
                    A[i, idx(ny, nx)] = -1
                b[i] = (
                    (grad_x[y, x-1] if x > 0 else 0) - (grad_x[y, x] if x < w-1 else 0)
                    + (grad_y[y-1, x] if y > 0 else 0) - (grad_y[y, x] if y < h-1 else 0)
                )
    solution = spsolve(A.tocsr(), b)
    return solution.reshape((h, w))

def poisson_blend_seidel(source, target, mask, offset=(0, 0), max_iter=5000, tol=1e-3):
    source_f = source.astype(np.float32)
    target_f = target.astype(np.float32)
    mask_f = (mask > 127).astype(np.float32)
    result = target_f.copy()
    h, w, c = source.shape
    tx, ty = offset
    
    mask_f = mask_f[ty:ty+h, tx:tx+w]

    for ch in range(c):
        src_channel = source_f[..., ch]
        trg_channel = result[..., ch]
        
        grad_x = np.pad(np.diff(src_channel, axis=1), ((0, 0), (0, 1)), mode='constant')[:, :-1]
        grad_y = np.pad(np.diff(src_channel, axis=0), ((0, 1), (0, 0)), mode='constant')[:-1, :]

        f = trg_channel[ty:ty+h, tx:tx+w].copy()

        for it in range(max_iter):
            f_prev = f.copy()
            
            f[1:-1, 1:-1] = 0.25 * (
                f[1:-1, :-2] + f[1:-1, 2:] + f[:-2, 1:-1] + f[2:, 1:-1]
                - (grad_x[1:-1, 1:] - grad_x[1:-1, :-1])
                - (grad_y[1:, 1:-1] - grad_y[:-1, 1:-1])
            )
            
            diff = np.linalg.norm(f - f_prev) / (np.linalg.norm(f_prev) + 1e-12)
            if diff < tol:
                print(f"[Channel {ch}] Converged at iteration {it}, diff {diff:.6f}")
                break

        result[ty:ty+h, tx:tx+w, ch] = np.where(mask_f, np.clip(f, 0, 255), trg_channel[ty:ty+h, tx:tx+w])

    return result.astype(np.uint8)