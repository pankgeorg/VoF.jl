module VoF

using WaterLily
using StaticArrays: SVector, MMatrix

export AlphaField, step_alpha!, VoFFlow, step_vof!, build_initial_L,
       step_vof_mules!, interior, viscosity,
       curvature!, csf_force!, surface_tension

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
                 — pass `ν=viscosity(vof)` to `Flow`/`Simulation` (a closure
                 wrapping this field; WaterLily reads it on the fly)
  * `L`        — face-staggered Poisson L = μ₀ / ρ_face
                 — pass `pois_ctor = flow -> MultiLevelPoisson(flow.p, vof.L, flow.σ)`
  * `r, Φ`     — workspace buffers for `WaterLily.transport!`

After each WaterLily step, call `step_vof!(vof, sim)` to:
  1. advect α with the just-projected velocity
  2. refresh ν_eff from the new α
  3. refresh L (= 1/ρ_face) from the new α
  4. set the Poisson coefficient via `WaterLily.density_coefficient!`
     (L = μ₀·(1/ρ)_face, then diagonal + multigrid refresh)

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
    # CSF surface-tension workspace (smoothed α and cell curvature).
    _st_αs :: Sf
    _st_κ  :: Sf
end

"""
    interior(vof::VoFFlow) -> CartesianIndices

Indices of the interior (non-ghost) cells of `vof.α`. Sized `grid_size`.
Use to iterate physical cells without touching the one-cell ghost layer.
"""
@inline interior(vof::VoFFlow) =
    CartesianIndices(ntuple(d -> 2:size(vof.α, d) - 1, ndims(vof.α)))

"""
    viscosity(vof::VoFFlow)

Closure `I -> ν[I]` over the effective-viscosity field, for passing to
`Flow(...; ν=viscosity(vof))` / `Simulation(...; ν=viscosity(vof))`.

WaterLily reads the effective viscosity on the fly through this closure
(no separate stored array on the `Flow`). It wraps `vof.ν` by reference,
so each in-place `_refresh_ν!` is seen by the next step — the same
contract as passing the array directly, which WaterLily no longer
accepts.
"""
@inline viscosity(vof::VoFFlow) = let ν = vof.ν
    I -> @inbounds ν[I]
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
        similar(α), similar(α), similar(α), similar(α),
        similar(α), similar(α), similar(α), similar(α),
        similar(L), similar(L), similar(L),
        similar(α), similar(α),
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
  3. refresh `vof.L` (= 1/ρ_face) from the new α (face-staggered)
  4. set the Poisson coefficient `L = μ₀·(1/ρ)_face` and refresh the solver
     via `WaterLily.density_coefficient!` (μ₀ from `sim.flow.μ₀`, so immersed
     bodies stay consistent with the density jump)

Call this after `sim_step!(sim)`. The Flow must have been constructed with
`ν = viscosity(vof)` and the Poisson with L matched to `vof.L` (see VoFFlow docstring).
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
    # 3. refresh 1/ρ_face
    _refresh_L!(vof; perdir=perdir)
    # 4. set the Poisson coefficient L = μ₀·(1/ρ)_face and refresh the
    # solver (diagonal + multigrid restriction). Folding in the measured
    # μ₀ keeps immersed bodies consistent with the density jump — the
    # previous plain copy of vof.L dropped μ₀ entirely.
    WaterLily.density_coefficient!(sim.pois, sim.flow.μ₀, vof.L; perdir=perdir)
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
# CSF surface tension (Brackbill, Kothe & Zemach 1992)
# ----------------------------------------------------------------------------
#
# f_st = σ · κ · ∇α  as a volumetric momentum source, with the curvature
# κ = -∇·(∇α_s/|∇α_s|) evaluated on a SMOOTHED colour function α_s
# (Brackbill's recommendation — the raw algebraic α is too noisy for
# second derivatives). The sharp ∇α in the force localizes it to the
# interface band; for a closed interface the discrete force sums to ~0
# (∮ κ n̂ ds = 0), so no spurious net momentum is injected.

# K passes of a 1/2-self + 1/2-neighbour-mean Jacobi smoother. Reads α,
# writes αs; uses κ as the ping-pong buffer.
function _smooth_α!(αs::AbstractArray{T,D}, tmp, α; passes::Int = 4) where {T,D}
    Ng = size(α)
    αs .= α
    half = T(0.5); w = half / (2D)
    interior = CartesianIndices(ntuple(d -> 2:Ng[d]-1, D))
    for _ in 1:passes
        tmp .= αs
        @inbounds for I in interior
            s = zero(T)
            for d in 1:D
                δd = WaterLily.δ(d, I)
                s += tmp[I+δd] + tmp[I-δd]
            end
            αs[I] = half * tmp[I] + w * s
        end
    end
    return αs
end

# |∇αs| at the d-face of cell I (normal component exact, tangential from
# averaged central differences) — same stencil as `_compression_flux`.
@inline function _face_grad_mag(αs::AbstractArray{T,D}, d, I) where {T,D}
    Im = I - WaterLily.δ(d, I)
    @inbounds gd = αs[I] - αs[Im]
    g2 = gd * gd
    @inbounds for k in 1:D
        k == d && continue
        δk = WaterLily.δ(k, I)
        gk = (αs[Im+δk] - αs[Im-δk] + αs[I+δk] - αs[I-δk]) / 4
        g2 += gk * gk
    end
    return gd, sqrt(g2)
end

"""
    curvature!(vof::VoFFlow; passes=2) -> vof._st_κ

Cell-centred interface curvature `κ = -∇·(∇α_s/|∇α_s|)` from the
smoothed colour function (`passes` Jacobi smoothing sweeps). κ is left
zero away from the interface (where `|∇α_s|` vanishes). For a 2D water
disk of radius R (α=1 inside), κ ≈ +1/R.
"""
function curvature!(vof::VoFFlow{T}; passes::Int = 4) where T
    α  = vof.α
    αs = _smooth_α!(vof._st_αs, vof._st_κ, α; passes)
    κ  = vof._st_κ
    fill!(κ, zero(T))
    D  = ndims(α)
    Ng = size(α)
    ϵg = T(1e-6)
    # κ[I] = -Σ_d ( n̂_d(face I+δd) - n̂_d(face I) ), faces indexed as in
    # WaterLily (face d of cell I sits between I-δd and I).
    @inbounds for I in CartesianIndices(ntuple(d -> 3:Ng[d]-2, D))
        s = zero(T)
        for d in 1:D
            δd = WaterLily.δ(d, I)
            g⁻, m⁻ = _face_grad_mag(αs, d, I)
            g⁺, m⁺ = _face_grad_mag(αs, d, I + δd)
            n⁻ = m⁻ > ϵg ? g⁻ / m⁻ : zero(T)
            n⁺ = m⁺ > ϵg ? g⁺ / m⁺ : zero(T)
            s += n⁺ - n⁻
        end
        κ[I] = -s
    end
    return κ
end

# --- Height-function curvature (Popinet, JCP 228, 2009; reimplemented from
# the paper, not from any GPL code) -------------------------------------------
#
# For an interface cell, build water-thickness columns H_t = Σ_k α along
# the axis closest to the interface normal (window ±3 cells), one column
# per tangential offset t. The curvature of the thickness function is
# orientation-free: a water bump and a water trough get opposite signs
# automatically, matching the `-∇·n̂` convention of `curvature!`
# (water disk → κ=+1/R, air bubble → κ=−1/R).
#
# A column is VALID only if, oriented along decreasing α, it runs from
# full (α>0.95) to empty (α<0.05) inside the window — i.e. it crosses
# the interface exactly once. Cells with any invalid column fall back
# to the smoothed-CSF estimate, so `curvature!(vof; method=:height)`
# is always defined everywhere `:smoothed` is.

# thickness column at cell I, direction axis `d`, sign `sd` (+e toward air),
# tangential offset `off` (a CartesianIndex displacement). Returns (H, valid).
@inline function _hf_column(α::AbstractArray{T,D}, I, d, sd, off, win) where {T,D}
    base = I + off
    Ng = size(α)
    # window bounds check
    lo = base - win * sd * WaterLily.δ(d, base)
    hi = base + win * sd * WaterLily.δ(d, base)
    for (P, lim) in ((lo, Ng), (hi, Ng))
        for k in 1:D
            (1 <= P.I[k] <= lim[k]) || return zero(T), false
        end
    end
    H = zero(T)
    @inbounds for k in -win:win
        H += α[base + k * sd * WaterLily.δ(d, base)]
    end
    @inbounds αw = α[base - win * sd * WaterLily.δ(d, base)]   # water end
    @inbounds αa = α[base + win * sd * WaterLily.δ(d, base)]   # air end
    valid = (αw > T(0.95)) & (αa < T(0.05))
    return H, valid
end

"""
    curvature!(vof; method=:smoothed, passes=4) -> vof._st_κ

Interface curvature. `method = :smoothed` (default) is the Brackbill
smoothed-`∇·n̂` estimate; `method = :height` uses Popinet-style
height-function columns at interface cells (2nd-order, much lower
scatter on sharp interfaces), falling back to the smoothed value where
the column construction is invalid (normal too oblique within the ±3
window, multiple crossings, or near the domain edge).
"""
function curvature!(vof::VoFFlow{T}, ::Val{:height}; passes::Int = 4) where T
    α = vof.α
    κ = curvature!(vof; passes)            # smoothed baseline + fallback
    D = ndims(α)
    Ng = size(α)
    win = 3
    @inbounds for I in CartesianIndices(ntuple(d -> 2:Ng[d]-1, D))
        # near-interface cells: significant central α-gradient. (An
        # α-range test misses sharp interfaces entirely — with a binary
        # step every cell is exactly 0 or 1, yet the force samples κ on
        # both sides of the jump.)
        gmax = zero(T); d = 1
        for k in 1:D
            g = (α[I + WaterLily.δ(k, I)] - α[I - WaterLily.δ(k, I)]) / 2
            if abs(g) > gmax
                gmax = abs(g); d = k
            end
        end
        gmax >= T(0.1) || continue
        gd = (α[I + WaterLily.δ(d, I)] - α[I - WaterLily.δ(d, I)]) / 2
        sd = gd < 0 ? 1 : -1                 # +e points toward air (α decreasing)
        if D == 2
            t = d == 1 ? 2 : 1
            H₋, v1 = _hf_column(α, I, d, sd, -WaterLily.δ(t, I), win)
            H₀, v2 = _hf_column(α, I, d, sd,  zero(WaterLily.δ(t, I)), win)
            H₊, v3 = _hf_column(α, I, d, sd,  WaterLily.δ(t, I), win)
            (v1 & v2 & v3) || continue
            H′ = (H₊ - H₋) / 2
            H″ = H₊ - 2H₀ + H₋
            q = 1 + H′^2
            κ[I] = -H″ / (q * sqrt(q))
        else
            t1, t2 = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)
            δ1 = WaterLily.δ(t1, I); δ2 = WaterLily.δ(t2, I)
            ok = true
            Hs = MMatrix{3,3,T}(undef)
            for j in -1:1, i in -1:1
                h, v = _hf_column(α, I, d, sd, i * δ1 + j * δ2, win)
                Hs[i+2, j+2] = h
                ok &= v
            end
            ok || continue
            Hx  = (Hs[3,2] - Hs[1,2]) / 2
            Hy  = (Hs[2,3] - Hs[2,1]) / 2
            Hxx = Hs[3,2] - 2Hs[2,2] + Hs[1,2]
            Hyy = Hs[2,3] - 2Hs[2,2] + Hs[2,1]
            Hxy = (Hs[3,3] - Hs[3,1] - Hs[1,3] + Hs[1,1]) / 4
            q = 1 + Hx^2 + Hy^2
            κ[I] = -(Hxx * (1 + Hy^2) + Hyy * (1 + Hx^2) - 2Hxy * Hx * Hy) /
                   (q * sqrt(q))
        end
    end
    return κ
end
curvature!(vof::VoFFlow, ::Val{:smoothed}; passes::Int = 4) = curvature!(vof; passes)

"""
    csf_force!(flow, vof::VoFFlow, σ; passes=4)

Add the Brackbill CSF surface-tension **acceleration** `σ·κ·∇α / ρ_face`
to `flow.f`. WaterLily's momentum residual is kinematic (`conv_diff!`
works in `u` and `ν`, gravity enters as an acceleration), so the
volumetric force `σκ∇α` must be divided by the local density; the face
`1/ρ` is read from `vof.L` — the *same* discretization the pressure
projection uses, which keeps the capillary pressure jump and the
projection consistent.

`σ` is in cell units consistent with the `ρ_w`/`ρ_a` passed to
`VoFFlow`: `σ_cell = σ_phys / U_ref²` when ρ keeps its physical
numeric value (the ΔX from κ and the 1/ρ make up the rest).

Use through WaterLily's `udf` hook:

    st! = VoF.surface_tension(vof, σ_cell)
    sim_step!(sim; udf = st!)
"""
function csf_force!(flow, vof::VoFFlow{T}, σ; passes::Int = 4,
                    method::Symbol = :smoothed) where T
    κ = curvature!(vof, Val(method); passes)
    α = vof.α
    invρf = vof.L          # face 1/ρ, refreshed by step_vof!/_mules! each step
    f = flow.f
    σT = T(σ)
    D  = ndims(α)
    Ng = size(α)
    @inbounds for d in 1:D
        for I in CartesianIndices(ntuple(k -> k == d ? (3:Ng[k]-1) : (2:Ng[k]-1), D))
            Im = I - WaterLily.δ(d, I)
            ∂α = α[I] - α[Im]                  # sharp gradient at the face
            iszero(∂α) && continue
            κf = (κ[I] + κ[Im]) / 2
            f[I, d] += σT * κf * ∂α * invρf[I, d]
        end
    end
    return f
end

"""
    surface_tension(vof::VoFFlow, σ; passes=4) -> udf closure

Convenience wrapper: returns `(flow, t; kwargs...) -> csf_force!(flow,
vof, σ)` for passing as `sim_step!(sim; udf = ...)`.
"""
surface_tension(vof::VoFFlow, σ; passes::Int = 4, method::Symbol = :smoothed) =
    (flow, t; kwargs...) -> csf_force!(flow, vof, σ; passes, method)

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
flux is computed with `λ_HO` (vanLeer by default) plus an
interface-compression flux `c_α·|u_f|·n̂·α(1-α)` (interFoam-style;
`c_α=1` default, `c_α=0` disables), the upwind flux provides the
monotone base, and a per-face Zalesak limiter tightens λ_face to keep
each cell's α inside its local extremum envelope.

After advecting α, refreshes `vof.ν` and `vof.L` exactly as `step_vof!`
does.
"""
# Interface-compression face flux (interFoam's cAlpha mechanism,
# reimplemented from the description in Berberović et al., Phys. Rev. E
# 79, 2009 — never from OpenFOAM source):
#
#     Φc[I,j] = c_α · |u_f| · n̂_j · α_f(1-α_f)
#
# where n̂ = ∇α/|∇α| at the face (normal component exact, tangential
# from averaged central differences) and α_f is the face mean. The term
# transports α *up-gradient* (anti-diffusion along the interface
# normal); α(1-α) confines it to interface cells, and the Zalesak
# envelope in step 5 keeps the result bounded — FCT applied to
# high-order + compression. Without it plain MULES has no re-steepening
# mechanism and homogenizes over long runs (see the 2026-06-11 Phase-2
# gate run in ShipFlow.jl/RESULTS-damBreak.md).
@inline function _compression_flux(α::AbstractArray{T,D}, uf, c_α, j, I) where {T,D}
    Im = I - WaterLily.δ(j, I)
    @inbounds αf = (α[I] + α[Im]) / 2
    s = αf * (one(T) - αf)
    s <= zero(T) && return zero(T)          # only interface cells compress
    @inbounds gj = α[I] - α[Im]             # exact face-normal gradient
    g2 = gj * gj
    @inbounds for k in 1:D
        k == j && continue
        δk = WaterLily.δ(k, I)
        gk = (α[Im+δk] - α[Im-δk] + α[I+δk] - α[I-δk]) / 4
        g2 += gk * gk
    end
    g2 <= eps(T) && return zero(T)
    return c_α * abs(uf) * (gj / sqrt(g2)) * s
end

# `λ_HO::FH` forces specialization on the limiter function — a plain
# untyped kwarg is NOT specialized by Julia, so every ϕu(...,λ_HO) call
# in the face-flux loop would dynamically dispatch and box (566 KiB/call
# at N=64² vs 4 KiB specialized).
# `c_α` scales the interface-compression flux: 0 = plain MULES (old
# behaviour, diffusive over long runs), 1 = interFoam default.
function step_vof_mules!(vof::VoFFlow{T}, sim;
                         dt::Real = sim.flow.Δt[end-1],
                         λ_HO::FH = WaterLily.vanLeer,
                         c_α::Real = one(T),
                         perdir = ()) where {T, FH}
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
    cα_T = T(c_α)

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
                ΦH[I, j] = WaterLily.ϕu(j, I, α_old, uf, λ_HO) +
                           _compression_flux(α_old, uf, cα_T, j, I)
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
    WaterLily.density_coefficient!(sim.pois, sim.flow.μ₀, vof.L; perdir=perdir)
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
