using Test
using VoF
using WaterLily
using StaticArrays

@testset "VoF" begin

    @testset "VoFFlow construction" begin
        dims = (16, 16)
        vof = VoFFlow(dims;
            α₀ = (i,x) -> x[2] < 8 ? 1f0 : 0f0,
            ρ_w = 1000.0, ρ_a = 1.0,
            μ_w = 1e-3,   μ_a = 1.8e-5,
        )
        # α matches the IC
        @test vof.α[8, 2]  ≈ 1f0
        @test vof.α[8, 16] ≈ 0f0

        # ν_eff = μ/ρ ⇒ in water 1e-3/1000 = 1e-6; in air 1.8e-5/1 = 1.8e-5
        @test isapprox(vof.ν[8, 2],  Float32(1e-3/1000);   rtol=1e-4)
        @test isapprox(vof.ν[8, 16], Float32(1.8e-5/1.0);  rtol=1e-4)

        # L = 1/ρ_face ⇒ in water 1/1000, in air 1/1
        @test size(vof.L) == ((dims .+ 2)..., 2)
        @test isapprox(vof.L[8, 2,  1], Float32(1/1000); rtol=1e-3)
        @test isapprox(vof.L[8, 16, 2], Float32(1.0);    rtol=1e-3)
    end

    @testset "VoFFlow interface face uses harmonic mean" begin
        dims = (4, 4)
        # Sharp interface: water at j=1,2 (y < waterline); air at j=3,4.
        # Cell-centre y = j-1.5 → waterline at y=2 (between j=3 and j=4).
        vof = VoFFlow(dims;
            α₀ = (i,x) -> x[2] < 2 ? 1f0 : 0f0,
            ρ_w = 1000.0, ρ_a = 1.0,
            μ_w = 1e-3,   μ_a = 1.8e-5,
        )
        # j=4 in the y-direction is between cell (any, 3) [water] and cell (any, 4) [air].
        # Cell centre y for j=3 is 1.5 (water), for j=4 is 2.5 (air).
        # 1/ρ_face = 0.5(1/1000 + 1/1) = 0.5005
        L_face = vof.L[2, 4, 2]
        @test isapprox(L_face, Float32(0.5 * (1/1000 + 1)); rtol=1e-3)
    end

    @testset "step_vof! preserves clamping" begin
        # Build the smallest sensible coupled flow + Poisson + VoF system
        # and take a few steps. α must stay in [0,1].
        dims = (16, 16)
        vof = VoFFlow(dims;
            α₀ = (i,x) -> x[2] < 8 ? 1f0 : 0f0,
            ρ_w = 1000.0, ρ_a = 1.0,
            μ_w = 1e-3,   μ_a = 1.8e-5,
        )
        flow = WaterLily.Flow(dims, (0f0, 0f0);
            T = Float32,
            ν = viscosity(vof),
            g = (i, x, t) -> i == 2 ? -9.81f0 : 0f0,
        )
        # Build a minimal struct that pretends to be a Simulation —
        # step_vof! only reads sim.flow and sim.pois.
        L0 = copy(vof.L)
        pois = WaterLily.MultiLevelPoisson(flow.p, L0, flow.σ)
        sim = (flow=flow, pois=pois)

        for _ in 1:5
            step_vof!(vof, sim; dt = 0.01)
        end
        @test minimum(vof.α) ≥ 0f0
        @test maximum(vof.α) ≤ 1f0
    end


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

    @testset "_bc_α! periodic round-trip" begin
        # Fill α with a y-varying sinusoid and confirm that periodic
        # BC wraps it: ghost cell 1 should match interior cell N+1,
        # and ghost cell N+2 should match interior cell 2.
        Ny = 16
        dims = (8, Ny)
        vof = VoFFlow(dims;
            α₀ = (i, x) -> 0.5f0 + 0.5f0 * sin(2pi * x[2] / Ny),
            ρ_w = 1.0, ρ_a = 1.0,
            μ_w = 1e-3, μ_a = 1e-3,
        )
        # Scramble the ghost layer first to make sure _bc_α! restores
        # it (not the constructor IC).
        vof.α[:, 1]       .= -99f0
        vof.α[:, Ny + 2]  .= -99f0
        VoF._bc_α!(vof.α, (2,))
        for i in 2:dims[1] + 1
            @test vof.α[i, 1]      ≈ vof.α[i, Ny + 1]
            @test vof.α[i, Ny + 2] ≈ vof.α[i, 2]
        end
    end

    @testset "MULES preserves [0,1] under uniform translation" begin
        # +x translation of a sharp step at constant velocity. The
        # limiter must keep α bounded.
        dims = (32, 16, 16)
        vof = VoFFlow(dims;
            α₀ = (i, x) -> x[1] < 10 ? 1f0 : 0f0,
            ρ_w = 1.0, ρ_a = 1.0, μ_w = 1e-3, μ_a = 1e-3,
        )
        sim = WaterLily.Simulation(dims, (1f0, 0f0, 0f0), 1f0;
            T = Float32, ν = viscosity(vof), Δt = 0.25f0, ϵ = 1, U = 1f0,
        )
        sim.flow.u .= 0f0
        sim.flow.u[:, :, :, 1] .= 1f0
        for _ in 1:6
            step_vof_mules!(vof, sim; dt = 0.25f0)
        end
        @test minimum(vof.α) ≥ -1f-4
        @test maximum(vof.α) ≤ 1f0 + 1f-4
    end

    @testset "MULES compression sharpens and conserves" begin
        # A pre-smeared interface advected at uniform velocity. With the
        # interFoam-style compression flux (c_α=1) the interface must
        # stay at least as sharp as without it (c_α=0), while mass stays
        # exactly conserved and α bounded — the Zalesak envelope limits
        # the compression like any other anti-diffusive flux.
        dims = (64, 8)
        mk() = VoFFlow(dims;
            # water band 13 < x < 35 with ~6-cell smeared edges, zero at
            # both walls so the open boundaries exchange no mass (a
            # column touching x=0 would legitimately gain inflow mass
            # through the upwind boundary flux)
            α₀ = (i, x) -> clamp((8f0 - abs(x[1] - 24f0)) / 6f0 + 0.5f0, 0f0, 1f0),
            ρ_w = 1.0, ρ_a = 1.0, μ_w = 1e-3, μ_a = 1e-3,
        )
        mkflow() = (f = WaterLily.Flow(dims, (1f0, 0f0); T = Float32, Δt = 0.2f0);
                    f.u .= 0f0; f.u[:, :, 1] .= 1f0; f)
        width(α)  = count(c -> 0.01f0 < c < 0.99f0, @view α[2:dims[1]+1, 2:dims[2]+1])
        mass(α)   = sum(@view α[2:dims[1]+1, 2:dims[2]+1])

        vof0, vof1 = mk(), mk()
        sim0 = (flow = mkflow(), pois = WaterLily.MultiLevelPoisson(zeros(Float32, dims .+ 2), copy(vof0.L), zeros(Float32, dims .+ 2)))
        sim1 = (flow = mkflow(), pois = WaterLily.MultiLevelPoisson(zeros(Float32, dims .+ 2), copy(vof1.L), zeros(Float32, dims .+ 2)))
        m0 = mass(vof0.α)
        for _ in 1:20
            step_vof_mules!(vof0, sim0; dt = 0.2f0, c_α = 0)
            step_vof_mules!(vof1, sim1; dt = 0.2f0, c_α = 1)
        end
        @test mass(vof0.α) ≈ m0 rtol = 1f-5      # both conserve
        @test mass(vof1.α) ≈ m0 rtol = 1f-5
        @test minimum(vof1.α) ≥ -1f-4            # bounded with compression
        @test maximum(vof1.α) ≤ 1f0 + 1f-4
        @test width(vof1.α) ≤ width(vof0.α)      # at least as sharp
        @test width(vof1.α) ≤ 4 * dims[2]        # and genuinely thin (≤4 cells/row)
    end

    @testset "interior() helper hits only physical cells" begin
        dims = (8, 8, 8)
        vof = VoFFlow(dims;
            α₀ = (i, x) -> 0f0,
            ρ_w = 1.0, ρ_a = 1.0, μ_w = 1e-3, μ_a = 1e-3,
        )
        idx = VoF.interior(vof)
        @test size(idx) == dims
        for I in idx
            for d in 1:3
                @test 2 ≤ I[d] ≤ dims[d] + 1
            end
        end
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
