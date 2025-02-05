# %%
import jax.numpy as jnp
import jax.random as jr
from pymdp.priors import dirichlet_prior
from pymdp.envs.simplest import SimplestEnv
import unittest

class TestPriors(unittest.TestCase):
    
    def test_dirichlet_prior_expectation(self):
        """Test that Dirichlet expectations sum to 1 along the appropriate axis (ie are categorical distributions)"""
        
        # Initialize simplest environment
        key = jr.PRNGKey(0)
        env = TMazeEnv(batch_size=1)

        # Initialize random Dirichlet priors and their expectations
        pA, A = dirichlet_prior(env.params["A"], init="random", scale=10.0, learning_enabled=True, key=key)
        pB, B = dirichlet_prior(env.params["B"], init="random", scale=10.0, learning_enabled=True, key=key)
        pD, D = dirichlet_prior(env.params["D"], init="random", scale=10.0, learning_enabled=True, key=key)

        # Check if expectations sum to 1 (within numerical tolerance)
        tol = 1e-6
        
        # Get sums using JAX
        A_sums = A[0].sum(axis=1)
        B_sums = B[0].sum(axis=1)
        D_sums = D[0].sum(axis=1)

        # Check that all sums are close to 1 using JAX
        self.assertTrue(jnp.allclose(A_sums, jnp.ones_like(A_sums), rtol=tol))
        self.assertTrue(jnp.allclose(B_sums, jnp.ones_like(B_sums), rtol=tol))
        self.assertTrue(jnp.allclose(D_sums, jnp.ones_like(D_sums), rtol=tol))

if __name__ == '__main__':
    unittest.main()
