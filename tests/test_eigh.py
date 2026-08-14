import jax
import linox


def test_eigh():
    A = jax.random.normal(jax.random.PRNGKey(0), (4, 4))
    A = A @ A.T
    linop = linox.kron(linox.Matrix(A), linox.Matrix(A))

    _ev, _evec = linox.leigh(linop)


test_eigh()
