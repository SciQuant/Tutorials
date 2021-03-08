# In this tutorial, you will learn:
#
#    - How to price American/Bermudan style derivatives using a Monte Carlo expected value
#      estimate through the Longstaff-Schwartz method.
#
# ## Introduction
#

# ## Setup
#
# The first step is to load the pertinent libraries for the tutorial:

using UniversalPricing
using UnPack

# ## Pricing
#
# Consider the same dividend-free stock ``S`` with Black-Scholes-Merton dynamics under the
# risk-neutral measure that we have already discussed in previous tutorials:

S0 = SVector{1}(36.)
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

T = 1.
Δt = 1/50
dynamics = [:S => S]
params = (r = 0.06, σ = 0.2)
ds = DynamicalSystem(f, g, dynamics, params)
sol = montecarlo(ds, T, 10_000; alg=UniversalDynamics.EM(), dt=Δt, seed=1);

# The Longstaff Schwartz algorithm expects, for any type of american/bermudan style
# derivative, the following arguments:
#
#    - the excercise value , i.e. a function that computes the net payment seen by the
#      option holder at each excercise date,
#    - a function for discounting between excercise dates,
#    - the regressors for the hold value fitting, and
#    - the set of excercise dates.

# The excercise value at each excercise time ``t`` for an american put option with strike
# ``K`` is given by:

# ```math
# U(t) = \max \left( K - S(t), 0 \right).
# ```

# The excercise value is provided as a function:

function U(u, p, t, Tenors=nothing, n=nothing)
    @unpack _securities_ = p
    @unpack _S_ = _securities_

    ## unpack strike parameter
    @unpack r, K = p

    S = remake(_S_, u)

    return max(K - S(t), 0)
end;

# The excercise value function reads as follows: compute the excercise value given the
# simulation value `u` at excercise time `t` with the aid of parameters `p`. The other
# arguments, `Tenors` and `n` are optional and refer to `t = Tenors[n]`, where `Tenors` is
# the array containing all the excercise dates and 0 and `n` the index in such array. For
# some products it is useful to have such information, as we will see for Callable Libor
# Exotics.
#
# If we decide to work with the spot measure ``Q^B``, the discount factor is used for
# discounting. In this particular case, we assume a flat rate model as interest rate model:

function D(u, p, t, T, Tenors=nothing, n=nothing, n′=nothing)
    ## unpack risk-free interest rate parameter
    @unpack r = p
    return exp(-r * (T - t))
end;

# Note that this function has an additional argument, `n′`, such that `T = Tenors[n′]`.
#
# We now have to define the regressors for each early excercise date. For this particular
# case we use the stock price as regressor:

function R(u, p, t, Tenors=nothing, n=nothing)
    @unpack _securities_ = p
    @unpack _S_ = _securities_
    S = remake(_S_, u)
    return S(t)
end;

# The excercise dates can be provided as a Tenor structure, where all dates must be sorted
# and include 0 or as a `τ` structure, where we must only provide the ``Δt_i`` between each
# early excercise date:

τ = fill(Δt, Int(T/Δt));

# We can now compute the expectation that yields to the fair price of the callable option:

params = (ds.params..., K = 40.)
price = callable_product_valuation(sol, params, U, D, R, τ=τ)
