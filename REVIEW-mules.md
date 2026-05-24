# Code review — MULES implementation

Review of the `step_vof_mules!` work in `src/VoF.jl` (commits `5e78ecf`,
`13350c1`, `62c8f79`) plus the related Kelvin investigation
(`ShipFlow.jl/scripts/wigley_kelvin.jl`).

Performed by a fresh subagent against the committed code; findings
recorded verbatim below.

## High-priority findings

- **[HIGH] Periodic boundary not refreshed.** When MULES runs with
  `perdir=(2,)` (as the Kelvin driver does), the y-direction ghost
  cells are never periodically wrapped. `step_vof_mules!` neither calls
  `WaterLily.perBC!(α_old, perdir)` nor refreshes ghosts after the IC
  fill. Result: periodic-y is silently broken — works only as long as
  the IC happens to be y-uniform, but waves developing in the
  simulation will diverge at the periodic seam. (Compare `_refresh_L!`
  which does call `WaterLily.BC!(L, ..., perdir)`.)

- **[HIGH] Architectural mismatch with WaterLily's transport pattern.**
  The boundary-skip fix in `62c8f79` (`Im.I[j] != 1` guards) is correct
  but reinvents what WaterLily already does via
  `scalar_lowerBoundary!`/`scalar_upperBoundary!`. The cleaner fix is to
  re-impose ghost α via a `BC_α!` call at the top of each step, then
  keep the MULES kernel index-symmetric. Current approach silently
  violates discrete conservation across the boundary face (ghost is
  treated as an infinite reservoir — fine for inflow, wrong for
  periodic).

- **[HIGH] 12 allocations per call (`step_vof_mules!`).**
  `α_old, ΦU, ΦH, α_UD, A, α_max, α_min, P_pos, P_neg, R_pos, R_neg,
  λ_face` are all heap-allocated every step. Lift them onto the
  `VoFFlow` struct as pre-allocated workspace; combined with fusing the
  five separate grid passes (Φ-compute + α_UD-update; P_pos/P_neg +
  α_max/α_min + λ-application), the call cost should drop from 263 ms
  to ~120 ms (≈40 % gain).

- **[HIGH] Five separate grid passes with identical index sets.**
  Steps 1+2 (compute Φ, update α_UD) and 6+7 (build λ, apply correction)
  are independently fuseable. At minimum, fuse those two pairs.

## Medium-priority findings

- **[MEDIUM] `_local_extrema!` uses a 3-point per-direction stencil**
  (5/7 neighbours in 2D/3D), not the full 3ᴰ−1 corner stencil that
  Marquez Damián 2013 specifies. This is more permissive (matches the
  OpenFOAM `MULES` convention) so not strictly wrong, but worth
  flagging if precision becomes the gate.

- **[MEDIUM] `A = ΦH .- ΦU` allocates a fresh `(Ng…, D)` array each
  call.** Boundary handling already sets `ΦH = ΦU` there, so `A=0` at
  boundary faces — consistent, but the explicit `A` array is wasted.
  Compute `af = ΦH[I,j] - ΦU[I,j]` inline in the P_pos/λ loops to skip
  this allocation entirely.

- **[MEDIUM] `wigley_kelvin.jl` inflow mask is mis-sized.** The mask
  `i < hull_xc - L_c` excludes only the leftmost ~3 cells (at default
  domain), not the inflow region the comment suggests. After
  `62c8f79` the drainage band is gone, so this is harmless cosmetics,
  but the mask intent is mis-sized.

## Low-priority findings

- **[LOW] `ε = eps(T)` in the R_pos / R_neg denominators is
  `≈1.2 × 10⁻⁷` at Float32.** Legitimate `dt · P_pos` values can be
  smaller than that during quiescent flow, causing the
  `max(dt·P, ε)` clamp to artificially trip the limiter. Use
  `ε = 1e-12` (in T-precision) instead.

- **[LOW] Hull footprint mask is the axis-aligned bbox** rather than
  the Wigley parabolic silhouette. Cosmetic — leaves a grey rectangle
  instead of the actual hull plan view at bow/stern.

- **[LOW] Kelvin reference lines emanate from `x_stern`** but the
  dominant wave source for a Wigley hull at low-to-mid Fr is the bow.
  Lines under-shoot the actual wedge by ~L_c. Visualisation only.

- **[LOW] Colormap clipping at `quantile(|η|, 0.90) × 2`** is fragile
  if wake noise dominates — would clip a legitimate bow wave. Use
  `quantile(|η|, 0.99)` directly. The `max(ηmax, 0.5)` floor is fine.

## Top recommendation

> Lift all 12 work buffers onto `VoFFlow`, add `BC_α!`/`perBC!(α,
> perdir)` at the start of `step_vof_mules!`, and replace the explicit
> `Im.I[j] != 1` guards with a structurally cleaner "write only to
> interior" index set. That fixes both the perdir bug and the alloc
> cost in one refactor.

This is the right next step for MULES. Filed as a follow-up; not done
in this session.
