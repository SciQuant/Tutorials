# In this tutorial, you will learn:
#
#    - How to declare dynamics with arbitrary coefficients,
#    - How to declare dynamics with known coefficients,
#    - How to declare dynamical systems.
#
# # Introduction
#
# In **UniversalDynamics** a dynamics represents continuous time, ``D``-dimensional Ito
# Systems of Stochastic Differential Equations (SDEs) in a time span ``\mathbb{I} =
# \left[t_0, T \right]``:
#
# ```math
# d\vec{u}(t) = f(t, \vec{u}(t)) \cdot dt + g(t, \vec{u}(t)) \cdot d\vec{W}(t), \quad \vec{u}(t_0) = \vec{u}_0,\\
# ```
#
# with drift coefficient ``f \colon \mathbb{I} \times \mathbb{R}^D \rightarrow
# \mathbb{R}^D``, diffusion coefficient ``g \colon \mathbb{I} \times \mathbb{R}^D
# \rightarrow \mathbb{R}^{D \times M}``, ``M``-dimensional driving Wiener correlated or
# uncorrelated process ``\vec{W}(t)`` and initial condition ``\vec{u}_0``.
#
# The library represents dynamics by means of two main types, `SystemDynamics` and
# `ModelDynamics`. The first case refers to SDEs with arbitrary coefficients while the
# second case is a supertype for dynamics with known coefficients that are already
# implemented in the library.
#
# Finally, as discussed in the [documentation](https://sciquant.github.io/UniversalDynamics.jl/dev/ad/dynamics.html#Introduction),
# the general expression of a dynamics can be reduced depending on the noise type.
#
# # Setup
#
# The first step is to load the library:

using UniversalDynamics

# Always take into account that any object documentation can be inspected using `?` followed
# by the object name. For example:

# ```
# # Type ? to enter help mode
# help?> SystemDynamics
# ```
#
# # System Dynamics
#
# A System of Stochastic Differential Equations with arbitrary coefficients is declared by
# its initial condition or state, initial time, noise type and noise correlations. For
# example, a three dimensional system with scalar noise is given using:

x0 = rand(3)
x = SystemDynamics(x0; noise=ScalarNoise())

# Notice that since `x0 isa Vector`, the system is defined in its in-place form. Otherwise,
# if we would like to work with the out of place version, `x0` must be defined as a
# `SVector`.
#
# On the other hand, diagonal noise is the default noise type, for example:

y0 = rand(2)
y = SystemDynamics(y0)

# Lastly, the non-diagonal noise case with correlated noises can be given using:

z0 = rand(1)
ρ = [exp(abs(i - j) / 2 * log(0.663)) for i in 1:3, j in 1:3]
z = SystemDynamics(z0; noise=NonDiagonalNoise(3), ρ=ρ)

# Notice that coefficients for these kind of dynamics haven't been described yet. They are
# provided in a further step, only if needed. This will become more clear after reading the
# following sections.

# # Model Dynamics
#
# A Model Dynamics refers to Stochastic Differential Equations that are known and fairly
# common in finance, such that it is worth having their coefficients implemented in the
# library. In this case, we need to provide additional parameters for the coefficients. For
# example, the well known One-Factor Vasicek model can be given as a One-Factor Short Rate
# Model of Affine type (see documentation
# [here](https://sciquant.github.io/UniversalDynamics.jl/dev/ir/short_rate_model.html#One-Factor-Affine-Model)
# for further details) with some assumptions, namely:
#    - ``κ``, ``θ`` and ``Σ`` as time independent parameters,
#    - ``α = 1`` and ``β = 0``.

r0 = rand(1)
κ(t) = 0.4363
θ(t) = 0.0613
Σ(t) = 0.1491
r = OneFactorAffineModelDynamics(r0, κ, θ, Σ, one, zero)

# As already mentioned, coefficients for these kind of dynamics are already implemented in
# the library. It is possible to access to them by calling either the `drift` or `difussion`
# functions on their in-place or out-of-place versions, which are dispatched depending on
# the parameters type. For example, for the in-place version we need to call
# `coefficient!(du, u, p, t)`, such that:

du = similar(state(r))
u = state(r)
p = UniversalDynamics.parameters(r)
t = UniversalDynamics.initialtime(r)

UniversalDynamics.drift!(du, u, p, t)
du
#-
du = similar(UniversalDynamics.noise_rate_prototype(r))
UniversalDynamics.diffusion!(du, u, p, t)
du

# One might wonder why do we need to implement model coefficients in the library. For this
# particular example, implementing fast coefficients is easy. However, for Multi-Factor
# models or other models, computations get more complicated and it is useful to aid the user
# with fast and traceble functions.

# # Dynamical Systems

# Once we have defined different dynamics, we can group them in a unique system by declaring
# a dynamical system (checkout the
# [documentation](https://sciquant.github.io/UniversalDynamics.jl/dev/ad/dynamicalsystem.html)
# for more details).

dynamics = [:x => x, :y => y, :z => z, :r => r]
ds = DynamicalSystem(dynamics)

# This dynamical system has relevant information, such as its state:

state(ds)

# the initial time:

UniversalDynamics.initialtime(ds)

# the correlations between noises:

cor(ds)

# and the noise rate prototype, which represents the prototype that the difussion function
# either modify in-place or return:

UniversalDynamics.noise_rate_prototype(ds)