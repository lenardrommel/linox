
import jax
import jax.numpy as jnp
from linox.api import eigh
from linox.operators import Kronecker, Matrix
from linox.operators.kron import topk_eigh


def test_kronecker_topk_eigh():
    print("Testing Kronecker top-k eigh...")
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    
    # Create random PSD matrices
    n1, n2 = 4, 4
    A_mat = jax.random.normal(k1, (n1, n1))
    A_mat = A_mat @ A_mat.T
    B_mat = jax.random.normal(k2, (n2, n2))
    B_mat = B_mat @ B_mat.T
    
    A = Matrix(A_mat)
    B = Matrix(B_mat)
    op = Kronecker(A, B)
    
    # Full dense kronecker
    full_kron = jnp.kron(A_mat, B_mat)
    true_vals, true_vecs = jnp.linalg.eigh(full_kron)
    
    # Sort descending
    true_vals = true_vals[::-1]
    true_vecs = true_vecs[:, ::-1]
    
    k = 5
    
    # Current behavior check (might fallback to exact/lanczos or work if topk_eigh is already hooked up or we call it directly)
    # The existing topk_eigh in kron.py exists but spectral.eigh likely doesn't call it specifically yet.
    
    print("Calling current topk_eigh directly...")
    try:
        vals, _vecs, _info = topk_eigh(op, k=k, largest=True)
        print("topk_eigh returned successfully.")
        print("Vals shape:", vals.shape)
        # Verify values
        print("True top k vals:", true_vals[:k])
        print("Computed vals:", vals)
        if jnp.allclose(vals, true_vals[:k], rtol=1e-4):
            print("Eigenvalues match!")
        else:
            print("Eigenvalues DO NOT match.")
            
    except Exception as e:
        print(f"topk_eigh failed: {e}")

    print("\nCalling linox.eigh(op, k=k)...")
    try:
        vals_eigh, _vecs_eigh = eigh(op, k=k)
        print("linox.eigh returned successfully.")
        print("Vals shape:", vals_eigh.shape)
        print("Computed vals:", vals_eigh)
        if jnp.allclose(vals_eigh, true_vals[:k], rtol=1e-4):
            print("Eigenvalues match via linox.eigh!")
        else:
            print("Eigenvalues DO NOT match via linox.eigh.")
    except Exception as e:
        print(f"linox.eigh failed: {e}")

if __name__ == "__main__":
    test_kronecker_topk_eigh()
