import jax.numpy as jnp
import jax.random as jr
from pymdp.priors import dirichlet_prior
from pymdp.envs.simplest import SimplestEnv
from pymdp.envs.tmaze import TMaze
import unittest

""" Unit Tests for prior initialization
__author__: Lancelot Da Costa
"""

class TestPriors(unittest.TestCase):
    
    def _check_categorical_distributions(self, env, key, tol = 1e-6):
        """Helper method to check if Dirichlet expectations sum to 1 for a given environment within given tolerance"""
        # Initialize random Dirichlet priors and their expectations
        pA, A = dirichlet_prior(env.params["A"], init="random", scale=10.0, learning_enabled=True, key=key)
        pB, B = dirichlet_prior(env.params["B"], init="random", scale=10.0, learning_enabled=True, key=key)
        pD, D = dirichlet_prior(env.params["D"], init="random", scale=10.0, learning_enabled=True, key=key)
        
        # Get sums using JAX
        for i in range(len(A)):  # Loop through all observation modalities
            A_sums = A[i].sum(axis=1)
            self.assertTrue(jnp.allclose(A_sums, jnp.ones_like(A_sums), rtol=tol))
        
        for i in range(len(B)):  # Loop through all state transition factors
            B_sums = B[i].sum(axis=1)
            self.assertTrue(jnp.allclose(B_sums, jnp.ones_like(B_sums), rtol=tol))
        
        for i in range(len(D)):  # Loop through all initial state factors
            D_sums = D[i].sum(axis=1)
            self.assertTrue(jnp.allclose(D_sums, jnp.ones_like(D_sums), rtol=tol))
    
    def test_dirichlet_prior_expectation(self):
        """Test that Dirichlet expectations sum to 1 along the appropriate axis (ie are categorical distributions)"""
        
        key = jr.PRNGKey(0)
        keys = jr.split(key, 2)  # Get two different keys for the two environments
        
        # Test SimplestEnv
        simple_env = SimplestEnv(batch_size=1)
        self._check_categorical_distributions(simple_env, keys[0])
        
        # Test TMaze
        tmaze_env = TMaze(batch_size=1)
        self._check_categorical_distributions(tmaze_env, keys[1])

if __name__ == '__main__':
    unittest.main()
