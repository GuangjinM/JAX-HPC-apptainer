import jax
import jax.numpy as jnp

# Enable double precision for scientific computing
jax.config.update("jax_enable_x64", True)

print("===== JAX environment =====")
print("JAX version:", jax.__version__)
print("Default backend:", jax.default_backend())
print("Devices:", jax.devices())

print("\n===== FP64 GPU test =====")

x = jnp.ones((2000, 2000), dtype=jnp.float64)
y = x @ x

# Force synchronization so that the GPU computation actually completes
y.block_until_ready()

print("Calculation completed.")
print("dtype:", y.dtype)
print("shape:", y.shape)
print("device:", y.device)
