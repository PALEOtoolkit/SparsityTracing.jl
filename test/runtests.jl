import ADelemtree; const AD = ADelemtree # short name for convenience

using SparseArrays
using LinearAlgebra  # for I

using Test
# using BenchmarkTools



@testset "ADelem" begin

    @testset "createviewassign" begin

        x = [24.0, 42.0]   

        x_ad = AD.create_advec(x)

        x_jac = AD.jacobian(x_ad, length(x))
        @test x_jac == Matrix(I, 2, 2)

        a_ad = x_ad[1]
        b_ad = x_ad[2]

        @test a_ad.val   == 24.0
        @test AD.deriv(a_ad, 2) == [1, 0]
        @test b_ad.val   == 42.0
        @test AD.deriv(b_ad, 2) == [0, 1]

        ax_ad = view(x_ad, 1)
        bx_ad = view(x_ad, 2)

        @test ax_ad[].val   == 24.0
        @test AD.deriv(ax_ad[], 2) == [1, 0]
        @test bx_ad[].val   == 42.0
        @test AD.deriv(bx_ad[], 2) == [0, 1]



        y_ad = similar(x_ad)
        ay_ad = view(y_ad, 1)
        by_ad = view(y_ad, 2)

        ay_ad[] = ax_ad[]
        by_ad[] = ax_ad[]
        @test ay_ad[].val   == 24.0
        @test AD.deriv(ay_ad[], 2) == [1, 0]
        @test by_ad[].val   == 24.0
        @test AD.deriv(by_ad[], 2) == [1, 0]

        by_ad[] = 25
        @test by_ad[].val   == 25.0
        @test AD.deriv(by_ad[], 2) == [0, 0]
        @test ax_ad[].val   == 24.0
        @test AD.deriv(ax_ad[], 2) == [1, 0]

    end

    @testset "arithmetic" begin

        x = [24.0, 42.0]   

        x_ad = AD.create_advec(x)

        a_ad = x_ad[1]
        b_ad = x_ad[2]

        # addition
        c_ad = a_ad + b_ad
        @test c_ad.val   == 24.0+42.0
        @test AD.deriv(c_ad, 2) == [1, 1]

        c_ad = a_ad + 2
        @test c_ad.val   == 24.0+2
        @test AD.deriv(c_ad, 2) == [1, 0]

        # check handling of ad variables with no derivatives
        x_no_ad = copy(x_ad)
        for i = 1:length(x_no_ad)
            x_no_ad[i] = AD.value(x_no_ad[i])
        end
        @test AD.deriv(x_no_ad[1], 2) == [0, 0]
        @test AD.deriv(x_no_ad[2], 2) == [0, 0]
        a_no_ad = x_no_ad[1]
        b_no_ad = x_no_ad[2]
        c_ad = a_no_ad + b_ad
        @test c_ad.val   == 24.0+42.0
        @test AD.deriv(c_ad, 2) == [0, 1]  # no deriv wrt x_ad[1]
        c_ad = a_ad + b_no_ad
        @test c_ad.val   == 24.0+42.0
        @test AD.deriv(c_ad, 2) == [1, 0]  # no deriv wrt x_ad[2]
        c_ad = a_no_ad + b_no_ad
        @test c_ad.val   == 24.0+42.0
        @test AD.deriv(c_ad, 2) == [0, 0]  # no deriv

        # unary minus
        c_ad = -a_ad
        @test c_ad.val   == -24.0
        @test AD.deriv(c_ad, 2) == -[1, 0]

        # subtraction
        c_ad = a_ad - b_ad
        @test c_ad.val   == 24.0-42.0
        @test AD.deriv(c_ad, 2) == [1, -1]

        c_ad = a_ad - 2
        @test c_ad.val   == 24.0-2
        @test AD.deriv(c_ad, 2) == [1, 0]

        c_ad = 2 - a_ad
        @test c_ad.val   == 2-24.0
        @test AD.deriv(c_ad, 2) == -[1, 0]

        #multiplication
        c_ad = a_ad*b_ad
        @test c_ad.val   == 24.0*42.0
        @test AD.deriv(c_ad, 2) == [42, 24]
    
        c_ad = a_ad*2
        @test c_ad.val   == 24.0*2
        @test AD.deriv(c_ad, 2) == [1, 0]*2

        # division
        c_ad = a_ad/b_ad
        @test c_ad.val   == 24.0/42.0
        # println(c_ad)
        @test AD.deriv(c_ad, 2) == [1/42, -24/42^2]
    
        c_ad = a_ad/2
        @test c_ad.val   == 24.0/2
        @test AD.deriv(c_ad, 2) == [1, 0]/2

        c_ad = 2/a_ad
        @test c_ad.val   == 2/24.0
        @test AD.deriv(c_ad, 2) == [-2/24^2, 0]

        # power
        c_ad = a_ad^2
        @test c_ad.val   == 24.0^2
        @test AD.deriv(c_ad, 2) == [1*24*2, 0]
       
        # max
        x_max = max.(x_ad, 30)
        @test x_max[1].val == 30
        @test x_max[2].val == 42.0
        @test AD.deriv(x_max[1], 2) == [0, 0]
        @test AD.deriv(x_max[2], 2) == [0, 1]
    end

    @testset "matmul" begin
        B = SparseArrays.sprand(10,10,0.5)
        C = AD.create_advec(ones(10))

        # A = AD.create_advec(zeros(10))
        A = B*C
        Ajac = AD.jacobian(A, length(C))

        @views A_view = B*C
        Ajac_view = AD.jacobian(A_view, length(C))
        @test Ajac_view == Ajac

        A_range = AD.create_advec(zeros(10))
        A_range .= 0.0

        A_range[2:5]  = B[2:5, :]*C
        A_range_jac = AD.jacobian(A_range, length(A_range))

        A_range .= 0.0
        @views A_range[2:5]  = B[2:5, :]*C  # fails - fills zeros in derivatives? (-> @views is removing sparsity ??)
        A_range_views_jac = AD.jacobian(A_range, length(A_range))
        @test A_range_views_jac == A_range_jac
        @test_broken nnz(A_range_views_jac) == nnz(A_range_jac) # fails -  A_range_view_jac has zero elements stored

        A_range .= 0.0
        mul!(view(A_range, 2:5), B[2:5, :], C)
        A_range_mul_jac = AD.jacobian(A_range, length(A_range))
        @test A_range_mul_jac == A_range_jac
        @test nnz(A_range_mul_jac) == nnz(A_range_jac)

        A_range .= 0.0
        @views mul!(A_range[2:5], B[2:5, :], C) # fails - fills zero in derivatives?
        A_range_mul_views_jac = AD.jacobian(A_range, length(A_range))
        @test A_range_mul_views_jac == A_range_jac
        @test_broken nnz(A_range_mul_views_jac) == nnz(A_range_jac) # fails -  A_range_mul_views_jac has zero elements stored

        # Looks like this is a symptom of a more fundamental problem with (lack of) views into sparse matrices,
        # and reverting to slow dense path:
        # https://stackoverflow.com/questions/58699267/julia-view-of-sparse-matrix
        # https://github.com/JuliaLang/julia/issues/21796

    end

    @testset "rober" begin
        function rober(du,u,p)
            y₁,y₂,y₃ = u
            k₁,k₂,k₃ = p
            du[1] = -k₁*y₁+k₃*y₂*y₃
            du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
            du[3] =  k₂*y₂^2
            nothing
        end

        function rober_jac(J,u,p)
            y₁,y₂,y₃ = u
            k₁,k₂,k₃ = p
            J[1,1] = k₁ * -1
            J[2,1] = k₁
            J[3,1] = 0
            J[1,2] = y₃ * k₃
            J[2,2] = y₂ * k₂ * -2 + y₃ * k₃ * -1
            J[3,2] = y₂ * 2 * k₂
            J[1,3] = k₃ * y₂
            J[2,3] = k₃ * y₂ * -1
            J[3,3] = 0
            nothing
        end 

        p = (0.04,3e7,1e4)

        u = [1.0, 2.0, 3.0]
        Janalytic = zeros(3, 3)
        rober_jac(Janalytic, u, p)

        @info "rober J analytic\n $Janalytic"

        u_ad = AD.create_advec(u)
        du_ad = similar(u_ad)
        rober(du_ad, u_ad, p)

        Jad = AD.jacobian(du_ad, length(du_ad))
        @info "rober J AD\n $Jad"

        @test Janalytic == Jad
    end
    
    return nothing
end
  
