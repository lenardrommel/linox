import jax.numpy as jnp

# Test what happens when we do row @ vec with various shapes
row = jnp.ones((3,))  # Shape (n1,)
vec1 = jnp.ones((3, 2))  # Shape (n1, k)
vec2 = jnp.ones((1, 3, 2))  # Shape (batch, n1, k)

print("row.shape:", row.shape)
print("vec1.shape:", vec1.shape)
print("row @ vec1:", (row @ vec1).shape)
print()

print("vec2.shape:", vec2.shape)
print("row @ vec2:", (row @ vec2).shape)
print()

# What the Kronecker product might pass
vec3 = jnp.ones((2, 3))  # After swapaxes: (k, n1)
print("vec3.shape:", vec3.shape)
try:
    result = row @ vec3
    print("row @ vec3:", result.shape)
except Exception as e:
    print("row @ vec3: ERROR -", e)
