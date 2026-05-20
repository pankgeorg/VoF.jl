module VoF

using WaterLily
using StaticArrays

export AlphaField, step_alpha!

"""
    AlphaField(grid_size; α₀=I->0, T=Float32, mem=Array)

Cell-centered colour function `α ∈ [0,1]` for two-phase VoF. `α₀` is an
initial-condition callable: either a function of `CartesianIndex` or a
function of the location `SVector`. Returned struct also carries the
workspace `r` (residual) and `Φ` (flux buffer) reused on each step.

# Example
```julia
α = AlphaField(dims;
    α₀ = (i,x) -> x[3] < 0 ? 1f0 : 0f0,  # water below z=0
)
```
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
        # default: all water
        α .= one(T)
    else
        _fill_ic!(α, α₀, T)
    end
    α  = α  |> mem
    r  = similar(α); fill!(r, zero(T))
    Φ  = similar(α); fill!(Φ, zero(T))
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
        # assume f(x::SVector)
        for I in CartesianIndices(α)
            α[I] = T(f(_ic_loc(I, T)))
        end
    end
end

"""
    step_alpha!(αf::AlphaField, u, Δt; λ=quick, perdir=())

Advance `αf.α` one explicit Euler step using `WaterLily.transport!` for
the spatial discretization. Clamps result back into [0,1] to bound
algebraic VoF.

This is a *minimal* algebraic VoF: pure advection by the flow velocity,
no compressive flux yet. MULES limiter and CSF surface tension come in
follow-up commits.
"""
function step_alpha!(αf::AlphaField{T}, u, Δt::Real;
                     λ=WaterLily.quick, perdir=()) where T
    WaterLily.transport!(αf.r, αf.α, u, αf.Φ; D_diff=zero(T), λ, perdir)
    @inbounds for I in CartesianIndices(αf.α)
        αf.α[I] = clamp(αf.α[I] + T(Δt) * αf.r[I], zero(T), one(T))
    end
    return αf.α
end

end # module
