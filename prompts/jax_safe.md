# Writing Safe JAX Code

## Core Principles

### 1. Use JAX Control Flow Primitives

❌ **Avoid:**

```python
for i in range(n):
    result = result + compute(i)
```

✅ **Use:**

```python
def body_fn(i, result):
    return result + compute(i)
result = jax.lax.fori_loop(0, n, body_fn, init_result)
```

### 2. Replace Conditionals with jnp.where()

❌ **Avoid:**

```python
if condition:
    result = value_a
else:
    result = value_b
```

✅ **Use:**

```python
result = jnp.where(condition, value_a, value_b)
```

### 3. Efficient Branch Selection

#### Option A: lax.switch (Preferred - Only Computes Selected Branch)

```python
# Map string to index at Python time
feature_map = {"min": 0, "max": 1, "mean": 2, "std": 3}
index = feature_map.get(feature, 4)

# Define branches as lambdas
branches = [
    lambda x: jnp.min(x),
    lambda x: jnp.max(x),
    lambda x: jnp.mean(x),
    lambda x: jnp.std(x),
    lambda x: jnp.zeros_like(x[0]),  # default
]

return lax.switch(index, branches, x)
```

#### Option B: Early Returns (When Parameter is Known at Python Time)

```python
# If 'mode' is a Python string (not traced), use regular conditionals
if mode == "mean":
    return jnp.mean(x)
elif mode == "var":
    return jnp.var(x)
elif mode == "std":
    return jnp.std(x)
else:
    return jnp.zeros_like(x[0])
```

#### Option C: Precompute All Branches (Least Efficient)

❌ **Avoid unless necessary:**

```python
# Computes ALL branches even though only one is needed
results = {
    "mean": jnp.mean(x),
    "var": jnp.var(x),
    "std": jnp.std(x),
}
return results.get(mode, jnp.mean(x))
```

### 4. Vectorize with vmap

❌ **Avoid:**

```python
for b in range(batch_size):
    for s in range(state_size):
        result[b, s] = compute(x[:, b, s])
```

✅ **Use:**

```python
compute_batch = jax.vmap(jax.vmap(compute, in_axes=1), in_axes=1)
result = compute_batch(x)
```

### 5. Per-Feature JIT Compilation

For large feature sets, JIT individual functions instead of one giant computation graph:

```python
# Pre-JIT each feature separately
jitted_features = {
    key: jax.jit(func) for key, func in feature_functions.items()
}

# Use fori_loop to iterate through features
def extract_feature(i, result):
    feat = jitted_features[keys[i]](x)
    return result.at[i].set(feat)

features = jax.lax.fori_loop(0, n_features, extract_feature, init_array)
```

## Key Benefits

- ✅ Compact XLA graphs → faster compilation
- ✅ Better GPU parallelization
- ✅ Type stability for XLA optimization
- ✅ No host-device synchronization overhead
