import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, r_squared

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    # This test should initially fail
    pass

def test_bootstrap_sample():
    """Test that bootstrap_sample works"""
    # Simple dataset: linear relationship y = 2*X + noise
    n = 200
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n, 1))
    y = 2*X[:, 0] + rng.normal(scale=0.5, size=n)
    
    # Add intercept column
    X_design = np.c_[np.ones(n), X]

    # Statistic: slope coefficient
    def slope_stat(X, y):
        # OLS closed form (X'X)^-1 X'y
        beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
        return beta[1]  # slope

    # Run bootstrap
    boot_stats = bootstrap_sample(X_design, y, slope_stat, n_bootstrap=500, random_state=42)
    
    # --- Assertions ---
    # 1. Check output shape
    assert boot_stats.shape == (500,), "Bootstrap output has wrong shape"
    
    # 2. Check variability
    assert boot_stats.std() > 0, "Bootstrap distribution has zero variance"
    
    # 3. Check mean is close to true slope (~2)
    mean_est = boot_stats.mean()
    assert np.isclose(mean_est, 2, atol=0.2), f"Bootstrap mean {mean_est:.3f} not close to 2"
    
    print("All tests passed ✅")
    print(f"Bootstrap mean slope: {mean_est:.3f}")
    print(f"Bootstrap std slope: {boot_stats.std():.3f}")

