# In this tutorial, you will learn:
#
#    - How to declare Short Rate Model dynamics correctly.
#
# ## Introduction
#
# We have already presented some short rate models in previous tutorials. In this tutorial
# we show more details but the general aspects have been already presented.
#
# ## Setup
#
# The first step is to load the pertinent libraries for the tutorial:

using UniversalDynamics
using UniversalPricing
using UnPack

# ## One-Factor Affine Short Rate Model
#
# The detailed documentation for such model is presented
# [here](https://sciquant.github.io/UniversalDynamics.jl/dev/ir/short_rate_model.html#One-Factor-Affine-Model).
# As you might see, this model involves practically all the most used models in practice,
# such as Vasicek, Cox-Ingersoll-Ross, Hull-White and Gaussian Short Rate (GSR). For
# example, the Vasicek model can be declared as:

r0 = rand(1)
κ(t) = 0.4363
θ(t) = 0.0613
Σ(t) = 0.1491
α(t) = one(t)
β(t) = zero(t)
r = OneFactorAffineModelDynamics(r0, κ, θ, Σ, α, β)

# The fact that ``κ``, ``θ`` and ``Σ`` are time independent parameters, ``α = 1``, ``β =
# 0``, ``\xi_0 = 0`` and ``\xi_1 = 1`` makes the One-Factor Affine model collapse to a
# Vasicek short rate model.
#
# ## Multi-Factor Affine Short Rate Model
#
# The detailed documentation for such model is presented
# [here](https://sciquant.github.io/UniversalDynamics.jl/dev/ir/short_rate_model.html#Multi-Factor-Affine-Model).
# It can be seen that this model generalizes the One-Factor case in a beautiful way. The
# only task left to the user is to define parameters. Take into account that those
# parameters have to be declared differently depending if we want to work with an in-place
# or out-of-place model version. This is because we want to avoid allocations. For example,
# the in-place version of a function that returns a matrix might be:

function ϰ!(u, t)
    u[1,1] = μ
    u[2,2] = ν
    u[3,1] = κ_rυ
    u[3,2] = -κ
    u[3,3] = κ
    return nothing
end

# while its out-of-place version:

function ϰ(t)
    return @SMatrix [
        μ     0 0
        0     ν 0
        κ_rυ -κ κ
    ]
end

# with known constants `μ`, `ν`, `κ_rυ` and `κ`.

# ## One-Factor Quadratic Short Rate Model
#
# The detailed documentation for such model is presented
# [here](https://sciquant.github.io/UniversalDynamics.jl/dev/ir/short_rate_model.html#One-Factor-Quadratic-Model).

# ## Multi-Factor Quadratic Short Rate Model
#
# The detailed documentation for such model is presented
# [here](https://sciquant.github.io/UniversalDynamics.jl/dev/ir/short_rate_model.html#Multi-Factor-Quadratic-Model).
# Note that the same considerations regarding in or out-of place coefficients applies for
# this short rate model.
#
# ## Money Market Account
#
# The current implementation of Short Rate models requires the declaration of the money
# market account as a separate process or dynamics. This is because we are insterested in
# solving its differential equation in the integrator, jointly with the short rate dynamics.
# In this context, if you are interested in using the money market account you must declare
# its dynamics:

## money market account dynamics (IIP case)
B = SystemDynamics([1.])

# and include it in the dynamics container for the `DynamicalSystem`.

# ## Basic Fixed Income Securities
#
# Interest rate models allows us to define all the basic fixed income securities. For the
# Short Rate Model, we can declare these objects simply using `FixedIncomeSecurities` inside
# any coefficient or payoff function, for example, for ``t ≤ T ≤ S``:

function test_coefficient_fis(du, u, p, t)
    @unpack _dynamics, _securities_ = p
    @unpack _r = _dynamics
    @unpack _r_, _B_ = _securities_

    T = 0.5
    S = 1.0

    r = remake(_r_, du, u, t)
    B = remake(_B_, du, u, t)

    IR = FixedIncomeSecurities(_r, r, B)

    ps = 22
    println(rpad("Spot rate: ", ps), IR.r(t))
    println(rpad("Money market account: ", ps), IR.B(t))
    println(rpad("Discount bond: ", ps), IR.P(t, T))
    println(rpad("Simple forward rate: ", ps), IR.L(t, T, S))
end

dynamics = [:r => r, :B => B]
ds = DynamicalSystem(dynamics)
du = similar(get_state(ds))
u = get_state(ds)
p = get_parameters(ds)
t = get_t0(ds)

test_coefficient_fis(du, u, p, t)

# For this dummy coefficient function we have been able to compute some basic fixed income
# securities that only needed information at the current time ``t``. On the other hand,
# computing the discount factor ``D(t, T)`` requires information at time ``T`` in the
# future, i.e. ``B(T)``. This implies that discount factors cannot be used in coefficient
# functions. However, they can be used in payoff function because in those cases we already
# have simulations up to time ``T`` or further, for example:

function f!(du, u, p, t)
    @unpack _dynamics, _securities_ = p
    @unpack _r = _dynamics
    @unpack _r_, _B_ = _securities_

    r = remake(_r_, du, u, t)
    B = remake(_B_, du, u, t)

    IR = FixedIncomeSecurities(_r, r, B)

    drift!(r.dx, r(t), get_parameters(_r), t)
    B.dx[] = IR.r(t) * B(t)

    return nothing
end

function g!(du, u, p, t)
    @unpack _dynamics, _securities_ = p
    @unpack _r = _dynamics
    @unpack _r_, _B_ = _securities_

    r = remake(_r_, du, u, t)
    B = remake(_B_, du, u, t)

    IR = FixedIncomeSecurities(_r, r, B)

    diffusion!(r.dx, r(t), get_parameters(_r), t)
    B.dx[] = zero(eltype(u))

    return nothing
end

ds = DynamicalSystem(f!, g!, dynamics, nothing)
sol = solve(ds, 1., alg=UniversalDynamics.EM(), dt=0.01, seed=1)

function test_payoff_fis(sol, p)
    @unpack _dynamics, _securities_ = p
    @unpack _r = _dynamics
    @unpack _r_, _B_ = _securities_

    T = 0.5
    S = 1.0

    r = remake(_r_, sol)
    B = remake(_B_, sol)

    IR = FixedIncomeSecurities(_r, r, B)

    ps = 22
    println(rpad("Spot rate: ", ps), IR.r(t))
    println(rpad("Money market account: ", ps), IR.B(t))
    println(rpad("Discount factor: ", ps), IR.D(t, T))
    println(rpad("Discount bond: ", ps), IR.P(t, T))
    println(rpad("Simple forward rate: ", ps), IR.L(t, T, S))
    println(rpad("Libor rate: ", ps), IR.L(T, S))
end

test_payoff_fis(sol, p)
