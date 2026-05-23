module VoF

using WaterLily
using StaticArrays

export AlphaField, step_alpha!, VoFFlow, step_vof!, build_initial_L

# ----------------------------------------------------------------------------
# Pure α-advection (v0.1) — kept for back-compat with the existing unit tests
# ----------------------------------------------------------------------------

"""
    AlphaField(grid_size; α₀=nothing, T=Float32, mem=Array)

Cell-centered colour function α ∈ [0,1] for two-phase VoF. Pure
advection by an externally-supplied velocity field (no momentum
coupling). See `VoFFlow` for the variable-density version.
"""
struct AlphaField{T, Sf<:AbstractArray{T}}
    α :: Sf
    r :: Sf
    Φ :: Sf
end

function AlphaField(grid_size::NTuple;
                    α₀=nothing,
                    T::Type=Float32, mem=Array)
    Ng = grid_size .+ 2
    α = zeros(T, Ng)
    if α₀ === nothing
        α .= one(T)
    else
        _fill_ic!(α, α₀, T)
    end
    α = α |> mem
    r = similar(α); fill!(r, zero(T))
    Φ = similar(α); fill!(Φ, zero(T))
    AlphaField{T, typeof(α)}(α, r, Φ)
end

@inline _ic_loc(I::CartesianIndex{D}, T) where D = SVector{D,T}(I.I .- 1.5)

function _fill_ic!(α::AbstractArray{T,D}, f, ::Type{T}) where {T,D}
    if hasmethod(f, Tuple{Int,AbstractVector})
        for I in CartesianIndices(α)
            α[I] = T(f(0, _ic_loc(I, T)))
        end
    elseif hasmethod(f, Tuple{CartesianIndex})
        for I in CartesianIndices(α)
            α[I] = T(f(I))
        end
    else
        for I in CartesianIndices(α)
            α[I] = T(f(_ic_loc(I, T)))
        end
    end
end

"""
    step_alpha!(αf::AlphaField, u, Δt; λ=quick, perdir=())

Advance `αf.α` one explicit Euler step using `WaterLily.transport!`.
Clamps to [0,1]. Pure advection, no momentum coupling.
"""
function step_alpha!(αf::AlphaField{T}, u, Δt::Real;
                     λ=WaterLily.quick, perdir=()) where T
    WaterLily.transport!(αf.r, αf.α, u, αf.Φ; D_diff=zero(T), λ, perdir)
    @inbounds for I in CartesianIndices(αf.α)
        αf.α[I] = clamp(αf.α[I] + T(Δt) * αf.r[I], zero(T), one(T))
    end
    return αf.α
end

# ----------------------------------------------------------------------------
# Variable-density VoF — the real two-phase coupling
# ----------------------------------------------------------------------------

"""
    VoFFlow(grid_size; α₀, ρ_w, ρ_a, μ_w, μ_a, T=Float32, mem=Array)

Two-phase VoF state for momentum-coupled simulations:

  * `α`        — cell-centered colour function (1 = water, 0 = air)
  * `ν`        — cell-centered effective kinematic viscosity (μ/ρ)_local
                 — pass this to `Flow(...; ν=vof.ν)` (PLAN 1 Hook 1)
  * `L`        — face-staggered Poisson L = μ₀ / ρ_face
                 — pass `pois_ctor = flow -> MultiLevelPoisson(flow.p, vof.L, flow.σ)`
  * `r, Φ`     — workspace buffers for `WaterLily.transport!`

After each WaterLily step, call `step_vof!(vof, sim)` to:
  1. advect α with the just-projected velocity
  2. refresh ν_eff from the new α
  3. refresh L (face-density-weighted) from the new α
  4. propagate the L change through MultiLevelPoisson levels

Gravity is applied via `Flow(...; g=(i,x,t)->(i==2 ? -g_phys : 0))` — uniform
acceleration (same on water and air). The variable-density physics emerges
from the L coefficient in the projection step.
"""
mutable struct VoFFlow{T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}}
    α   :: Sf
    r   :: Sf
    Φ   :: Sf
    ν   :: Sf
    L   :: Vf
    ρ_w :: T
    ρ_a :: T
    μ_w :: T
    μ_a :: T
end

function VoFFlow(grid_size::NTuple{D,Int};
                 α₀,
                 ρ_w::Real, ρ_a::Real,
                 μ_w::Real, μ_a::Real,
                 T::Type=Float32, mem=Array) where D
    Ng = grid_size .+ 2
    α = zeros(T, Ng)
    _fill_ic!(α, α₀, T)
    α  = α |> mem
    r  = similar(α); fill!(r, zero(T))
    Φ  = similar(α); fill!(Φ, zero(T))
    ν  = similar(α); fill!(ν, zero(T))
    L  = zeros(T, Ng..., D) |> mem
    vof = VoFFlow{T, typeof(α), typeof(L)}(
        α, r, Φ, ν, L,
        T(ρ_w), T(ρ_a), T(μ_w), T(μ_a),
    )
    _refresh_ν!(vof)
    _refresh_L!(vof)
    return vof
end

# Per-cell effective kinematic viscosity ν = μ_local / ρ_local.
function _refresh_ν!(vof::VoFFlow{T}) where T
    @inbounds for I in CartesianIndices(vof.α)
        a = vof.α[I]
        ρ = a * vof.ρ_w + (one(T) - a) * vof.ρ_a
        μ = a * vof.μ_w + (one(T) - a) * vof.μ_a
        vof.ν[I] = μ / ρ
    end
    return vof.ν
end

# Face-staggered Poisson coefficient L = 1/ρ_face.
# ρ_face[I,j] is the density at the j-face of cell I — the face shared with
# cell I-δ(j,I).  Use harmonic mean of 1/ρ (i.e. arithmetic mean of 1/ρ) for
# smoother transitions across the interface.
function _refresh_L!(vof::VoFFlow{T}; perdir=()) where T
    α   = vof.α
    L   = vof.L
    D   = ndims(α)
    Ng  = size(α)
    invρ_at(I) = let a = α[I]
        ρ = a * vof.ρ_w + (one(T) - a) * vof.ρ_a
        inv(ρ)
    end
    @inbounds for j in 1:D
        for I in CartesianIndices(α)
            iL = invρ_at(I)
            if I.I[j] > 1
                I_prev = CartesianIndex(ntuple(k -> k == j ? I.I[k] - 1 : I.I[k], D))
                iL_prev = invρ_at(I_prev)
                L[I, j] = T(0.5 * (iL + iL_prev))
            else
                L[I, j] = T(iL)
            end
        end
    end
    # Match WaterLily's μ₀ boundary convention: zero L at the wall ghost
    # layer so the Poisson enforces no-flux at no-slip walls.
    WaterLily.BC!(L, ntuple(_ -> zero(T), D), false, perdir)
    return vof.L
end

"""
    step_vof!(vof::VoFFlow, sim; dt=sim.flow.Δt[end-1], perdir=())

Post-step VoF update:
  1. advect `vof.α` by the current `sim.flow.u`
  2. refresh `vof.ν` from the new α (consumed by `conv_diff!` on the next step)
  3. refresh `vof.L` from the new α (face-staggered)
  4. propagate `vof.L` into the MultiLevelPoisson levels via `WaterLily.update!`

Call this after `sim_step!(sim)`. The Flow must have been constructed with
`ν = vof.ν` and the Poisson with L matched to `vof.L` (see VoFFlow docstring).
"""
function step_vof!(vof::VoFFlow{T}, sim;
                   dt::Real = sim.flow.Δt[end-1],
                   perdir=()) where T
    # 1. advect α
    WaterLily.transport!(vof.r, vof.α, sim.flow.u, vof.Φ;
                         D_diff=zero(T), perdir=perdir)
    @inbounds for I in CartesianIndices(vof.α)
        vof.α[I] = clamp(vof.α[I] + T(dt) * vof.r[I],
                         zero(T), one(T))
    end
    # 2. refresh ν_eff (in-place — array shared with flow)
    _refresh_ν!(vof)
    # 3. refresh L
    _refresh_L!(vof)
    # 4. propagate L into the Poisson levels.
    # MultiLevelPoisson has its own L array that is COPIED to levels[1].L
    # at construction. We need to copy vof.L over levels[1].L and then call
    # update! which restricts to coarser levels and refreshes D, iD.
    sim.pois.levels[1].L .= vof.L
    sim.pois.L           .= vof.L
    WaterLily.update!(sim.pois)
    return vof.α
end

"""
    build_initial_L(vof::VoFFlow, μ₀_flow)

Multiply the BDIM kernel `μ₀_flow` (from `Flow.μ₀`) element-wise by
`vof.L` (= 1/ρ_face) and return the resulting L array suitable for
`MultiLevelPoisson(flow.p, L, flow.σ; perdir)`.

In practice for damBreak (no body, μ₀_flow ≡ 1 in the interior), this
returns essentially a copy of `vof.L`.
"""
function build_initial_L(vof::VoFFlow{T}, μ₀_flow::AbstractArray{T}) where T
    L = similar(μ₀_flow)
    @inbounds for I in CartesianIndices(L)
        L[I] = μ₀_flow[I] * vof.L[I]
    end
    return L
end

end # module
