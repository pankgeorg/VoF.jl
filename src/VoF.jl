module VoF

using WaterLily
using StaticArrays: SVector

export AlphaField, step_alpha!, VoFFlow, step_vof!, build_initial_L,
       step_vof_mules!, interior

# Harmonic-mean of 1/ρ for the face Poisson coefficient; module-scope so the
# inner loop doesn't close over `vof`.
@inline _invρ(α, I, ρ_w::T, ρ_a::T) where T =
    inv(α[I] * ρ_w + (one(T) - α[I]) * ρ_a)

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

"""
    _fill_ic!(α, f, T)

Fill the α array from a user-supplied initial-condition function `f`.
Dispatches on the arity of `f` to support three common forms:

  * `f(i, x_cell)` — WaterLily-style with a component index
  * `f(I::CartesianIndex)` — index-only
  * `f(x_cell)` — position-only

`x_cell` is the cell-centre world coordinate (one-half cell inset from
the array origin, matching WaterLily's ghost convention).
"""
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

**Array layout.** `α`, `ν`, `r`, `Φ` are sized `grid_size .+ 2` with one
ghost cell per side; `L` is `(grid_size .+ 2)..., D`. The interior cells
are at `CartesianIndices(ntuple(d -> 2:N[d]-1, D))`. Use [`interior`](@ref)
to iterate the interior of any `VoFFlow` field.
"""
struct VoFFlow{T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}}
    α   :: Sf
    r   :: Sf
    Φ   :: Sf
    ν   :: Sf
    L   :: Vf
    ρ_w :: T
    ρ_a :: T
    μ_w :: T
    μ_a :: T
    # MULES workspace, allocated eagerly so step_vof_mules! has no first-call
    # cost and the struct stays immutable + type-stable.
    _mules_α_old :: Sf
    _mules_α_UD  :: Sf
    _mules_α_max :: Sf
    _mules_α_min :: Sf
    _mules_P_pos :: Sf
    _mules_P_neg :: Sf
    _mules_R_pos :: Sf
    _mules_R_neg :: Sf
    _mules_ΦU    :: Vf
    _mules_ΦH    :: Vf
    _mules_λface :: Vf
end

"""
    interior(vof::VoFFlow) -> CartesianIndices

Indices of the interior (non-ghost) cells of `vof.α`. Sized `grid_size`.
Use to iterate physical cells without touching the one-cell ghost layer.
"""
@inline interior(vof::VoFFlow) =
    CartesianIndices(ntuple(d -> 2:size(vof.α, d) - 1, ndims(vof.α)))

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
        similar(α), similar(α), similar(α), similar(α),
        similar(α), similar(α), similar(α), similar(α),
        similar(L), similar(L), similar(L),
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

"""
    _refresh_L!(vof; perdir=())

Recompute the face-staggered Poisson coefficient `L = 1/ρ_face` from
the current `vof.α`. `ρ_face[I, j]` is the density at the j-face of
cell I (the face shared with cell `I - δ(j,I)`); the harmonic mean of
`1/ρ` (= arithmetic mean of `1/ρ`) is used so the coefficient smooths
across the interface without picking up artefacts of the density jump.
Wall ghost faces are set to zero via `WaterLily.BC!`, matching the
μ₀ no-flux convention.
"""
function _refresh_L!(vof::VoFFlow{T}; perdir=()) where T
    α   = vof.α
    L   = vof.L
    D   = ndims(α)
    Ng  = size(α)
    ρ_w = vof.ρ_w; ρ_a = vof.ρ_a
    # Faces only between cells with index ≥ 2 along axis j; the ghost face
    # (I[j] == 1) is overwritten by BC! below — don't bother writing it.
    @inbounds for j in 1:D
        for I in CartesianIndices(ntuple(k -> k == j ? (2:Ng[k]) : (1:Ng[k]), D))
            iL      = _invρ(α, I, ρ_w, ρ_a)
            iL_prev = _invρ(α, I - WaterLily.δ(j, I), ρ_w, ρ_a)
            L[I, j] = (iL + iL_prev) / 2
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
            # Include cells that are not yet pinned to the deficit's
            # endpoint (positive deficit ⇒ accept cells with ai < 1;
            # negative deficit ⇒ accept cells with ai > 0). The previous
            # `zero(T) < ai < one(T)` form excluded any cell already
            # clamped exactly to 0 or 1, which under-redistributed.
            ε = eps(T)
            capacity = zero(T)
            @inbounds for I in interior
                ai = vof.α[I]
                if deficit > 0 ? ai < one(T) - ε : ai > ε
                    capacity += deficit > 0 ? (one(T) - ai) : ai
                end
            end
            if capacity > 0
                scale = deficit / capacity
                @inbounds for I in interior
                    ai = vof.α[I]
                    accept = deficit > 0 ? ai < one(T) - ε : ai > ε
                    if accept
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
    _push_L_to_pois!(vof, sim)
    return vof.α
end

# Push the just-refreshed L into all Poisson levels. MultiLevelPoisson's
# `L` was COPIED into `levels[1].L` at construction, so we re-copy on each
# step and let `update!` restrict to coarser levels.
function _push_L_to_pois!(vof::VoFFlow, sim)
    sim.pois.levels[1].L .= vof.L
    sim.pois.L           .= vof.L
    WaterLily.update!(sim.pois)
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
                    dt=sim.flow.Δt[end-1],
                    λ_HO=WaterLily.vanLeer, perdir=())

MULES α-advection step. Replaces `step_vof!` for cases where local
mass conservation matters (Kelvin waves, sloshing, …). The high-order
flux is computed with `λ_HO` (vanLeer by default), the upwind flux
provides the monotone base, and a per-face Zalesak limiter tightens
λ_face to keep each cell's α inside its local extremum envelope.

After advecting α, refreshes `vof.ν` and `vof.L` exactly as `step_vof!`
does.
"""
function step_vof_mules!(vof::VoFFlow{T}, sim;
                         dt::Real = sim.flow.Δt[end-1],
                         λ_HO = WaterLily.vanLeer,
                         perdir = ()) where T
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
    _local_extrema!(α_max, α_min, α_old, Ng)

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
    _push_L_to_pois!(vof, sim)
    return vof.α
end

"""
    _bc_α!(α, perdir) -> α

Apply boundary conditions to the scalar `α` field in-place. Periodic
directions (those listed in `perdir`) wrap via `WaterLily.perBC!`;
the remaining directions get zero-gradient Neumann (ghost cell copies
the first interior cell on each face).
"""
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

"""
    _local_extrema!(α_max, α_min, α, Ng) -> (α_max, α_min)

Fill `α_max[I]` and `α_min[I]` with the maximum and minimum of `α`
over `I` and its 2D neighbouring cells (3-point stencil per axis,
interior only). Used by `step_vof_mules!` to build the local envelope
for Zalesak limiting — the per-cell α target stays inside this
envelope on every step.
"""
function _local_extrema!(α_max, α_min,
                        α::AbstractArray{T,D},
                        Ng::NTuple) where {T,D}
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
