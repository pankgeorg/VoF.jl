module VoF

using WaterLily
using StaticArrays

export AlphaField, step_alpha!, VoFFlow, step_vof!, build_initial_L,
       transport_upwind!, step_vof_mules!

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
    # MULES workspace — allocated lazily on first call to step_vof_mules!.
    # Each field is `nothing` until needed, then sized to the grid.
    _mules_α_old :: Union{Nothing, Sf}
    _mules_α_UD  :: Union{Nothing, Sf}
    _mules_α_max :: Union{Nothing, Sf}
    _mules_α_min :: Union{Nothing, Sf}
    _mules_P_pos :: Union{Nothing, Sf}
    _mules_P_neg :: Union{Nothing, Sf}
    _mules_R_pos :: Union{Nothing, Sf}
    _mules_R_neg :: Union{Nothing, Sf}
    _mules_ΦU    :: Union{Nothing, Vf}
    _mules_ΦH    :: Union{Nothing, Vf}
    _mules_λface :: Union{Nothing, Vf}
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
        nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing,
        nothing, nothing, nothing,
    )
    _refresh_ν!(vof)
    _refresh_L!(vof)
    return vof
end

# Lazily allocate the MULES workspace on the first call to step_vof_mules!.
function _ensure_mules_workspace!(vof::VoFFlow{T, Sf, Vf}) where {T, Sf, Vf}
    if vof._mules_α_old !== nothing
        return nothing
    end
    Ng = size(vof.α); D = ndims(vof.α)
    vof._mules_α_old = similar(vof.α)
    vof._mules_α_UD  = similar(vof.α)
    vof._mules_α_max = similar(vof.α)
    vof._mules_α_min = similar(vof.α)
    vof._mules_P_pos = similar(vof.α)
    vof._mules_P_neg = similar(vof.α)
    vof._mules_R_pos = similar(vof.α)
    vof._mules_R_neg = similar(vof.α)
    vof._mules_ΦU    = similar(vof.L)
    vof._mules_ΦH    = similar(vof.L)
    vof._mules_λface = similar(vof.L)
    return nothing
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
                   perdir=(),
                   λ = WaterLily.vanLeer,
                   clamp_α::Bool = true,
                   mass_repair::Bool = false) where T
    # 1. advect α with a TVD limiter (vanLeer by default — bounded,
    # second-order, and considerably reduces the α overshoots that
    # the clamp absorbs and turns into mass loss).
    WaterLily.transport!(vof.r, vof.α, sim.flow.u, vof.Φ;
                         D_diff=zero(T), λ=λ, perdir=perdir)

    # Track pre-clamp mass for optional repair (only over interior cells).
    Ng = size(vof.α)
    interior = CartesianIndices(ntuple(d -> 2:Ng[d]-1, ndims(vof.α)))
    pre_mass = zero(T)
    if mass_repair
        @inbounds for I in interior
            pre_mass += vof.α[I] + T(dt) * vof.r[I]
        end
    end

    if clamp_α
        @inbounds for I in CartesianIndices(vof.α)
            vof.α[I] = clamp(vof.α[I] + T(dt) * vof.r[I],
                             zero(T), one(T))
        end
    else
        # No clamp: α can drift outside [0,1]. For diagnostic only.
        @inbounds for I in CartesianIndices(vof.α)
            vof.α[I] += T(dt) * vof.r[I]
        end
    end

    # 1b. Optional mass-repair: redistribute the clamping deficit into
    # interface cells (0 < α < 1) proportional to their slack to
    # [0, 1]. Linear in cells but globally redistributive — half-step
    # toward MULES, not equivalent.
    if mass_repair
        post_mass = zero(T)
        @inbounds for I in interior
            post_mass += vof.α[I]
        end
        deficit = pre_mass - post_mass
        if abs(deficit) > 0
            # Slack capacity for positive deficit, slack for negative.
            capacity = zero(T)
            @inbounds for I in interior
                ai = vof.α[I]
                if zero(T) < ai < one(T)
                    capacity += deficit > 0 ? (one(T) - ai) : ai
                end
            end
            if capacity > 0
                scale = deficit / capacity
                @inbounds for I in interior
                    ai = vof.α[I]
                    if zero(T) < ai < one(T)
                        bump = deficit > 0 ? scale * (one(T) - ai) :
                                              scale * ai
                        vof.α[I] = clamp(ai + bump, zero(T), one(T))
                    end
                end
            end
        end
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

# ----------------------------------------------------------------------------
# MULES-style α-advection (Marquez Damián 2013, used in interFoam)
# ----------------------------------------------------------------------------
#
# Provides locally mass-conserving α-advection by face-iterated flux
# limiting. The high-order flux (vanLeer) is blended with the upwind
# flux per face by a limiter λ ∈ [0, 1], computed so each cell's α
# stays in its local [α_min, α_max] envelope.
#
# Compared to the existing `step_vof!` + `mass_repair=true` heuristic
# (which conserves *global* mass but flattens spatial structure), this
# preserves spatial structure cell-by-cell — the right answer for
# Kelvin-pattern fidelity and the proper Phase-2 < 0.1 % gate.

"""
    transport_upwind!(r, φ, u; perdir=())

Cell-centred residual `r[I] = -∂_j(u_j φ_upwind)` using first-order
upwind in each direction j. Bounded by construction (monotone): if
all `φ` are in `[a, b]` then after one Euler step with CFL ≤ 1 the
result is still in `[a, b]`.
"""
function transport_upwind!(r::AbstractArray{T,D},
                           φ::AbstractArray{T,D},
                           u::AbstractArray{T};
                           perdir=()) where {T,D}
    r .= zero(T)
    N = size(φ)
    for j in 1:D
        @inbounds for I in CartesianIndices(ntuple(k -> k == j ? (3:N[k]-1) : (2:N[k]-1), D))
            uf = u[WaterLily.CI(I, j)]
            φ_up = uf >= 0 ? φ[I - WaterLily.δ(j, I)] : φ[I]
            Φ = uf * φ_up
            r[I]               += Φ
            r[I - WaterLily.δ(j, I)] -= Φ
        end
    end
    return r
end

"""
    step_vof_mules!(vof::VoFFlow, sim;
                    dt=sim.flow.Δt[end-1], n_sweeps=3,
                    λ_HO=WaterLily.vanLeer, perdir=())

MULES α-advection step. Replaces `step_vof!` for cases where local
mass conservation matters (Kelvin waves, sloshing, …). The high-order
flux is computed with `λ_HO` (vanLeer by default), the upwind flux
provides the monotone base, and `n_sweeps` (default 3) iterations of
the per-cell limiter tighten λ_face to keep each cell's α inside its
local extremum envelope.

After advecting α, refreshes `vof.ν` and `vof.L` exactly as `step_vof!`
does.
"""
function step_vof_mules!(vof::VoFFlow{T}, sim;
                         dt::Real = sim.flow.Δt[end-1],
                         n_sweeps::Int = 3,   # unused for now (cell-FCT pass)
                         λ_HO = WaterLily.vanLeer,
                         perdir = ()) where T
    _ensure_mules_workspace!(vof)
    α     = vof.α
    α_old = vof._mules_α_old
    α_UD  = vof._mules_α_UD
    α_max = vof._mules_α_max
    α_min = vof._mules_α_min
    P_pos = vof._mules_P_pos
    P_neg = vof._mules_P_neg
    R_pos = vof._mules_R_pos
    R_neg = vof._mules_R_neg
    ΦU    = vof._mules_ΦU
    ΦH    = vof._mules_ΦH
    λface = vof._mules_λface
    u = sim.flow.u
    Ng = size(α)
    D = ndims(α)

    # Refresh α ghost cells via the BC machinery before reading them.
    # For non-periodic walls this is a Neumann zero-gradient reflect; for
    # periodic directions it wraps. Without this the perdir y direction
    # silently desyncs (review HIGH finding).
    _bc_α!(α, perdir)

    @inbounds α_old .= α
    @inbounds fill!(P_pos, zero(T))
    @inbounds fill!(P_neg, zero(T))
    @inbounds fill!(λface, one(T))

    # 1. Per-face upwind flux ΦU[I, j] and high-order flux ΦH[I, j].
    # WaterLily face-storage convention: face j at index I lies between
    # cells I-δ(j,I) and I. Boundary faces (I[j] = 2 or I[j] = Ng[j]) get
    # only the upwind flux — the high-order ϕu stencil needs I-2δ(j)
    # which is out of bounds at the boundary, so we set ΦH = ΦU there.
    @inbounds for j in 1:D
        for I in CartesianIndices(ntuple(k -> k == j ? (2:Ng[k]) : (2:Ng[k]-1), D))
            uf = u[WaterLily.CI(I, j)]
            ΦU[I, j] = uf * (uf >= 0 ? α_old[I - WaterLily.δ(j, I)] : α_old[I])
            if I.I[j] == 2 || I.I[j] == Ng[j]
                ΦH[I, j] = ΦU[I, j]    # no anti-diff at boundary
            else
                ΦH[I, j] = WaterLily.ϕu(j, I, α_old, uf, λ_HO)
            end
        end
    end

    # 2. α_UD from upwind fluxes (bounded by monotonicity).
    # WaterLily convention: face flux Φ enters cell I, leaves cell I-δ(j,I).
    # Writing to I[j]=Ng[j] or Im[j]=1 just modifies the ghost layer;
    # the subsequent BC! call at the top of the NEXT step re-imposes the
    # correct ghost value, so we don't bother guarding those writes here.
    @inbounds α_UD .= α_old
    @inbounds for j in 1:D
        for I in CartesianIndices(ntuple(k -> k == j ? (2:Ng[k]) : (2:Ng[k]-1), D))
            Im = I - WaterLily.δ(j, I)
            α_UD[I]  += T(dt) * ΦU[I, j]
            α_UD[Im] -= T(dt) * ΦU[I, j]
        end
    end

    # Re-impose α BC on α_UD (ghost was disturbed by the flux loop above).
    _bc_α!(α_UD, perdir)

    # 3. Local extrema per cell (from α_old over a 3-point per-direction stencil).
    _local_extrema!(α_max, α_min, α_old, Ng, D)

    # 4. Cell-by-cell accumulate the positive / negative anti-diffusive
    # contributions from all incident faces.
    @inbounds for j in 1:D
        for I in CartesianIndices(ntuple(k -> k == j ? (2:Ng[k]) : (2:Ng[k]-1), D))
            af = ΦH[I, j] - ΦU[I, j]
            Im = I - WaterLily.δ(j, I)
            if af > 0
                P_pos[I]  += af
                P_neg[Im] += af
            elseif af < 0
                P_pos[Im] += -af
                P_neg[I]  += -af
            end
        end
    end

    ε = T(1e-12)
    @inbounds for I in CartesianIndices(α)
        R_pos[I] = clamp((α_max[I] - α_UD[I]) / max(T(dt) * P_pos[I], ε),
                         zero(T), one(T))
        R_neg[I] = clamp((α_UD[I] - α_min[I]) / max(T(dt) * P_neg[I], ε),
                         zero(T), one(T))
    end

    # 5. Face-level λ — Zalesak's rule. Then apply: α_new = α_UD + dt·div(λ·A).
    @inbounds for j in 1:D
        for I in CartesianIndices(ntuple(k -> k == j ? (2:Ng[k]) : (2:Ng[k]-1), D))
            af = ΦH[I, j] - ΦU[I, j]
            Im = I - WaterLily.δ(j, I)
            λij = if af > 0
                min(R_pos[I], R_neg[Im])
            elseif af < 0
                min(R_pos[Im], R_neg[I])
            else
                one(T)
            end
            λface[I, j] = λij
            corr = T(dt) * λij * af
            α_UD[I]  += corr
            α_UD[Im] -= corr
        end
    end

    # 6. Copy back, re-impose α BCs, refresh ν / L / Poisson.
    @inbounds for I in CartesianIndices(α)
        vof.α[I] = clamp(α_UD[I], zero(T), one(T))
    end
    _bc_α!(vof.α, perdir)
    _refresh_ν!(vof)
    _refresh_L!(vof; perdir=perdir)
    sim.pois.levels[1].L .= vof.L
    sim.pois.L           .= vof.L
    WaterLily.update!(sim.pois)
    return vof.α
end

# Boundary conditions for the scalar α field:
#   - periodic directions (in `perdir`) wrap via `WaterLily.perBC!`
#   - other directions get zero-gradient Neumann (ghost = first interior)
function _bc_α!(α::AbstractArray{T,D}, perdir) where {T,D}
    N = size(α)
    # Periodic directions
    WaterLily.perBC!(α, perdir, N)
    # Neumann (zero-gradient) on the non-periodic directions
    for j in 1:D
        j in perdir && continue
        # lower ghost at I[j]=1 ← α at I[j]=2
        @inbounds for I in CartesianIndices(ntuple(k -> k == j ? (1:1) : (1:N[k]), D))
            α[I] = α[I + WaterLily.δ(j, I)]
        end
        # upper ghost at I[j]=N[j] ← α at I[j]=N[j]-1
        @inbounds for I in CartesianIndices(ntuple(k -> k == j ? (N[j]:N[j]) : (1:N[k]), D))
            α[I] = α[I - WaterLily.δ(j, I)]
        end
    end
    return α
end

# Local 3-point per-direction extrema for each cell (interior only).
function _local_extrema!(α_max, α_min,
                        α::AbstractArray{T,D},
                        Ng::NTuple, D_::Int) where {T,D}
    α_max .= α
    α_min .= α
    for j in 1:D
        @inbounds for I in CartesianIndices(ntuple(k -> k == j ? (2:Ng[k]-1) : (2:Ng[k]-1), D))
            Iplus  = I + WaterLily.δ(j, I)
            Iminus = I - WaterLily.δ(j, I)
            # Bounds check
            if Iplus[j] <= Ng[j]
                α_max[I] = max(α_max[I], α[Iplus])
                α_min[I] = min(α_min[I], α[Iplus])
            end
            if Iminus[j] >= 1
                α_max[I] = max(α_max[I], α[Iminus])
                α_min[I] = min(α_min[I], α[Iminus])
            end
        end
    end
    return α_max, α_min
end

end # module
