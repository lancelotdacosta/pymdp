"""Minimal, self-contained demonstration that PyMDP's `spm_wnorm`
(over-)estimates Dirichlet information-gain compared with the true
KL divergence between Dirichlet distributions.  Running the script
produces a scatter-plot: each point is a random prior–observation
pair.  If the two measures agreed, points would lie on the diagonal.

PyMDP Code References:
- spm_wnorm implementation: pymdp/maths.py line 408
- calc_pA_info_gain usage: pymdp/control.py line 251  
- EFE calculation (standard): pymdp/control.py line 307 (adds +param_info_gain)
- EFE calculation (inductive): pymdp/control.py line 342 (adds -param_info_gain)
"""

import jax.numpy as jnp
from jax import random as jr
from jax.scipy.special import digamma, gammaln
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1.  Reference: true KL(Dir(α_post) || Dir(α_prior))
# ---------------------------------------------------------------------

def kl_dirichlet(alpha_prior: jnp.ndarray, alpha_post: jnp.ndarray) -> jnp.ndarray:
    """Closed-form KL between two Dirichlet distributions.
    
    Formula: KL(Dir(α_post) || Dir(α_prior)) = 
             log B(α_prior) - log B(α_post) + 
             Σ(α_post - α_prior)[ψ(α_post) - ψ(Σα_post)]
    """
    alpha_prior = jnp.maximum(alpha_prior, 1e-8)
    alpha_post  = jnp.maximum(alpha_post,  1e-8)

    def logB(a):
        """Log Beta function: log B(α) = Σlog Γ(α_i) - log Γ(Σα_i)"""
        return jnp.sum(gammaln(a)) - gammaln(jnp.sum(a))

    return (
        logB(alpha_prior) - logB(alpha_post)
        + jnp.sum((alpha_post - alpha_prior) * (digamma(alpha_post) - digamma(jnp.sum(alpha_post))))
    )

# ---------------------------------------------------------------------
# 2.  PyMDP's current surrogate: spm_wnorm · observation
#     (exact port of pymdp/maths.py:spm_wnorm + pymdp/control.py usage)
# ---------------------------------------------------------------------

def spm_wnorm_gain(alpha: jnp.ndarray, observation: jnp.ndarray, eps: float = 1e-16) -> jnp.ndarray:
    """PyMDP's parameter information gain calculation.
    
    From pymdp/maths.py line 408-410:
        norm = 1. / A.sum(axis=0)
        avg = 1. / (A + MINVAL)  
        wA = norm - avg
        
    From pymdp/control.py line 251:
        wa_m = spm_wnorm(pa_m) * (pa_m > 0.)
        return qo_m.dot(factor_dot(wa_m, qs_factors))
    """
    norm = 1.0 / alpha.sum()
    avg  = 1.0 / (alpha + eps)
    w    = norm - avg
    return jnp.dot(w * (alpha > 0), observation)

# ---------------------------------------------------------------------
# 3.  Generate sample pairs and plot
# ---------------------------------------------------------------------

def generate_samples(key: jr.PRNGKey, n: int = 200):
    """Generate random prior-observation pairs for comparison."""
    kl_vals, spm_vals = [], []
    for _ in range(n):
        key, k1, k2 = jr.split(key, 3)
        prior = jr.exponential(k1, (4,)) + 0.1  # strictly positive α
        observation = jnp.eye(4)[jr.choice(k2, 4)]  # single one-hot draw
        posterior = prior + observation

        kl_vals.append(kl_dirichlet(prior, posterior))
        spm_vals.append(spm_wnorm_gain(prior, observation))
    return jnp.array(kl_vals), jnp.array(spm_vals)

if __name__ == "__main__":
    kl, spm = generate_samples(jr.PRNGKey(0))

    # Scatter + y=x reference
    plt.scatter(spm, kl, alpha=0.7, s=40)
    limit = float(max(spm.max(), kl.max()))
    plt.plot([0, limit], [0, limit], "k--", lw=1)
    plt.xlabel("PyMDP  spm_wnorm  value")
    plt.ylabel("True  Dirichlet  KL")
    plt.title("PyMDP info-gain vs. correct KL (one-hot observations)")
    plt.tight_layout()
    plt.show()

    # Quick numeric summary
    ratio = spm / kl
    print(f"Average over-estimate: {ratio.mean():.1f}×  (min {ratio.min():.1f}×, max {ratio.max():.0f}×)")
