# In this tutorial, you will learn:
#
#    -  How to declare dynamics with arbitrary coefficients
#    -  How to declare dynamics with known coefficients
#    -  How to declare dynamical systems
#
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
## Setup
#
# The first step is to load the library:

using UniversalDynamics

## System Dynamics
#
# A System of Stochastic Differential Equations is declared by its initial condition,
# initial time, noise type and noise correlations, for example:

x = SystemDynamics(rand(1); noise=ScalarNoise())
y = SystemDynamics(rand(2); noise=DiagonalNoise(2), ρ=[1 0.3; 0.3 1])
z = SystemDynamics(rand(3); noise=NonDiagonalNoise(2), ρ=[1 0.2; 0.2 1])

## Model Dynamics
#
# A Model Dynamics refers to Stochastic Differential Equations that are known and fairly
# common in finance, such that is worth having their coefficients implemented in the
# library. In this case,  we need to provide the additional known parameters to the
# dynamics. For example, a One Factor Short Rate Model of Affine type that yields to the well
# known Vasicek model, can be given as:
r0 = rand(1)

ϰ = 0.4363
ϑ = 0.0613
σ = 0.1491

κ(t) = ϰ
θ(t) = ϑ
Σ(t) = σ

r = OneFactorAffineModelDynamics(r0, κ, θ, Σ, one, zero)



# group dynamics in a container
dynamics = [:x => x, :y => y, :z => z]

# compute dynamical system
ds = DynamicalSystem(dynamics)