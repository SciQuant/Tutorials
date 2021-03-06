# In this tutorial, you will learn:
#
#    - How to price European style derivatives using a Monte Carlo expected value estimate.
#
# ## Introduction
#
# Recall that the price of a European style derivative with time ``T`` payout ``V(T)`` is
# given by the expectation under an equivalent martingale measure. Specifically, given a
# deflator or numeraire ``N`` and assuming the existence of a equivalent martingale measure
# ``Q^N`` induced by ``N``, we have:

# ```math
# V(t) = N(t) \cdot E_t^{Q_N} \left[ \frac{V(T)}{N(T)} \right].
# ```

# The Monte Carlo method provides a numerical technique to compute the expectation of a
# random variable. In that context, we can proceed with the pricing of a given product once
# we have our simulations for the processes or dynamics that are involved and using the
# Strong Law of Large Numbers.

# ## Setup
#
# The first step is to load the pertinent libraries for the tutorial:

using UniversalPricing
using UnPack

# ## Pricing
#
# Consider the same dividend-free stock ``S`` with Black-Scholes-Merton dynamics under the
# risk-neutral measure that we have already discussed in previous tutorials:

S0 = @SVector ones(1)
S = SystemDynamics(S0)

function f(u, p, t)
    @unpack _securities_ = p
    @unpack _S_ = _securities_

    ## unpack risk-free rate parameter
    @unpack r = p

    S = remake(_S_, u, t)

    dS = r * S(t)

    return SVector{1}(dS)
end

function g(u, p, t)
    @unpack _securities_ = p
    @unpack _S_ = _securities_

    ## unpack volatility parameter
    @unpack σ = p

    S = remake(_S_, u, t)

    dS = σ * S(t)

    return SVector{1}(dS)
end

dynamics = [:S => S]
params = (r = 0.05, σ = 0.1)
ds = DynamicalSystem(f, g, dynamics, params)
sol = montecarlo(ds, 1., 10_000; seed=1);

# The payoff for a vanilla call option with strike ``K`` and maturity ``T`` is given by:

# ```math
# V(T) = \max \left( S(T) - K, 0 \right).
# ```

# Using the money market account ``B(t)`` as numeraire, we have:

# ```math
# V(0) = B(0) \cdot E_t^{Q} \left[ \frac{V(T)}{B(T)} \right] = E_t^{Q} \left[ D(0, T) \cdot V(T) \right]
# ```

# with ``D(t, T)`` the discount factor, which in the simple flat rate model is given by:

# ```math
# D(t, T) = \exp \left( -r \cdot (T - t) \right).
# ```

# In order to price such derivative, we first need to declare a function that computes the
# discounted payoff given the simulations and a set of parameters:

function discounted_payoff(sol, p)
    @unpack _securities_ = p
    @unpack _S_ = _securities_

    ## unpack strike and maturity parameters
    @unpack r, K, T = p

    S = remake(_S_, sol)

    return exp(-r * (T - 0)) * max(S(T) - K, 0)
end;

# We can now compute the expectation that yields to the fair price of the option:

params = (ds.params..., T = 1., K = 1.)
price = 𝔼(discounted_payoff, sol, params)