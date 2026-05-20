using Test
using VoF
using WaterLily
using StaticArrays

@testset "VoF" begin

    # Coordinate convention: AlphaField uses cell-center coords
    #     x = I.I .- 1.5
    # so for a grid of size N, interior cell-center x ranges over 0.5 to N-0.5.

    @testset "AlphaField construction + IC" begin
        dims = (16, 16)
        αf = AlphaField(dims;
            α₀ = (i, x) -> x[2] < 8 ? 1f0 : 0f0,    # water below y=8
        )
        @test size(αf.α) == dims .+ 2
        # Lower-y cells should be 1 (water), upper 0 (air).
        @test αf.α[8, 2] ≈ 1f0       # cell center y = 0.5
        @test αf.α[8, 16] ≈ 0f0      # cell center y = 14.5
    end

    @testset "still water: α unchanged after a step" begin
        dims = (16, 16)
        αf = AlphaField(dims;
            α₀ = (i, x) -> x[2] < 8 ? 1f0 : 0f0,
        )
        u = zeros(Float32, (dims .+ 2)..., 2)
        α0 = copy(αf.α)
        step_alpha!(αf, u, 0.1)
        @test α0 ≈ αf.α
    end

    @testset "uniform translation: front moves at the prescribed speed" begin
        # 2D step at x=16 (mid-domain) in a +x stream.
        dims = (64, 8)
        αf = AlphaField(dims;
            α₀ = (i, x) -> x[1] < 16 ? 1f0 : 0f0,
        )
        u = zeros(Float32, (dims .+ 2)..., 2)
        u[:, :, 1] .= 1f0
        Δt = 0.25f0
        nsteps = 8
        for _ in 1:nsteps
            step_alpha!(αf, u, Δt)
        end
        t = nsteps * Δt
        # In a +x stream, the front should advance by U·t = 2 cells.
        # Find the cell along y=4 where α first crosses below 0.5.
        row = @view αf.α[:, 4]
        i_front = something(findfirst(α -> α < 0.5f0, row), length(row))
        x_front = i_front - 1.5
        @test isapprox(x_front, 16 + t; atol = 1.5)
    end

    @testset "α stays in [0,1]" begin
        dims = (32, 32)
        αf = AlphaField(dims;
            α₀ = (i, x) -> (12 < x[1] < 20 && 12 < x[2] < 20) ? 1f0 : 0f0,
        )
        u = zeros(Float32, (dims .+ 2)..., 2)
        u[:, :, 1] .= 1f0
        for _ in 1:20
            step_alpha!(αf, u, 0.25)
        end
        @test minimum(αf.α) ≥ 0f0
        @test maximum(αf.α) ≤ 1f0
    end

    @testset "rotation conserves mass within bounded drift" begin
        # Solid-body rotation about (cx, cy). A blob rotates; total mass
        # may drift slightly because of upwind clamping at the edge of
        # the support — bound the drift to ~30% over a few steps.
        dims = (32, 32)
        cx, cy = 16f0, 16f0
        αf = AlphaField(dims;
            α₀ = (i, x) -> ((x[1] - cx)^2 + (x[2] - cy)^2 < 25) ? 1f0 : 0f0,
        )
        u = zeros(Float32, (dims .+ 2)..., 2)
        Ω = 0.05f0
        for I in CartesianIndices(u)
            x = I.I[1] - 1.5f0
            y = I.I[2] - 1.5f0
            if I.I[end] == 1
                u[I] = -Ω * (y - cy)
            elseif I.I[end] == 2
                u[I] =  Ω * (x - cx)
            end
        end
        m0 = sum(αf.α)
        for _ in 1:5
            step_alpha!(αf, u, 0.1)
        end
        m1 = sum(αf.α)
        @test isapprox(m1, m0; rtol = 0.3)
    end

end
