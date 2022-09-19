"""
Automatic differentiation with per-variable storage of derivative information (as a tree structure) in a new `Type` [`ADval`](@ref)` <: Real`.
This provides a way of detecting Jacobian (dy/dx) sparsity patterns for problems of form y = f(x)
where x and y are vectors.

 Notes:
  - Will fail if y = f(x) produces a tree with loops, eg by inverting a matrix to solve a linear system.
  - Very slow as each arithmetic operation generates a memory allocation (in order to form the derivative tree structure).  
    Speed is approx 10 Mop/s (on a ~4Ghz laptop core) so 100 - 1000x slower than `y = f(x)` with primitive types. Intent 
    is that the sparsity information is calculated once and then used to enable a separate fast AD calculator.
  - Also calculates derivative information (slowly).
  - The tree of derivative information grows without limit, so currently limited to cases `y = f(x)` where f does 
    not generate too many operations. It would be straightforward to fix this by limiting tree to size `~4*length(x)` 
    per element and simplifying it when this limit exceeded (as tree size = 2*length(x) per element is enough for a 
    dense Jacobian)
  - Uses DiffRules.jl to autogenerate operations.  
  - See ForwardDiff.Dual https://github.com/JuliaDiff/ForwardDiff.jl/blob/master/src/dual.jl 
    for examples of definitions required.
 
# Example usage:
    x_ad = SparsityTracing.create_advec(x)           # x_ad takes values from x,  derivatives set to identity
    y_ad = f(x_ad)                              # tree of derivative information is accumulated into each element of y_ad
    jac = SparsityTracing.jacobian(y_ad, length(x))  # walk trees and generate sparse Jacobian
 """
module SparsityTracing


using SparseArrays
using DiffRules

import SpecialFunctions

"base type for tree of derivative information"
abstract type AbstractDerivNode
end


"""
    ADval

Scalar variable type T with derivatives
"""
struct ADval{T<:Real} <: Real
    val::T
    derivnode::Union{Nothing, AbstractDerivNode}
end

"""
    create_advec(x::Vector{T}) -> Vector{ADval{T, length(x)}}

Create a vector of `ADval` scalars, each with value initialised from `x`
and `length(x)` derivatives initialised to identity (each element derivative 1 wrt itself and 0 elsewhere).
Usual case for an independent vector variable `x` (eg to use as input with `y = f(x)`).
 
# Examples:
 ```jldoctest; setup = :(import SparsityTracing)
julia> v_ad = SparsityTracing.create_advec([1.0, 5.0])
2-element Vector{SparsityTracing.ADval{Float64}}:
 SparsityTracing.ADval{Float64}(1.0, <derivnode>)
 SparsityTracing.ADval{Float64}(5.0, <derivnode>)

julia> x_ad, y_ad = v_ad;

julia> x_ad        # deriv is sparse vector
SparsityTracing.ADval{Float64}
  val=1.0
  deriv:
  [1]  =  1.0
   
julia> y_ad       # deriv is sparse vector
SparsityTracing.ADval{Float64}
  val=5.0
  deriv:
  [2]  =  1.0  

julia> SparsityTracing.jacobian(v_ad, 2)  # sparse 2x2 identity matrix.
2×2 SparseArrays.SparseMatrixCSC{Float64, Int64} with 2 stored entries:
 1.0   ⋅ 
  ⋅   1.0

julia> z_ad = x_ad*y_ad^2    # d(x*y^2)/dx = y^2 = 5^2 = 25,  d(x*y^2)/dy = 2*x*y = 2*1*5 = 10
SparsityTracing.ADval{Float64}
  val=25.0
  deriv:
  [1]  =  25.0
  [2]  =  10.0
```
"""
function create_advec(x::Vector{T}) where T
 
    n = length(x)
    ADvec = Vector{ADval{T}}(undef, n)
   
    for i = 1:n
        ADvec[i] = ADval{T}(x[i], DerivLeaf(i))        
    end

    return ADvec
end



###############################
# Construction and conversion #
###############################

"create from a non-AD value, set derivatives to zero"
ADval{T}(v) where {T} = ADval{T}(v, nothing)
ADval{T}(v::ADval{T}) where {T} = v

Base.convert(::Type{ADval{T}}, v) where {T} = ADval{T}(v)
Base.convert(::Type{ADval{T}}, v::Number) where {T} = ADval{T}(v)
Base.convert(::Type{ADval{T}}, d::ADval{T}) where {T} = d

Base.promote_rule(::Type{T}, ::Type{ADval{T}} ) where {T} = ADval{T}

"get value. NB: no automatic conversion to Float64 etc"
value(d::ADval) = d.val

#####################
# Generic functions #
#####################

Base.zero(d::ADval{T}) where {T}        = zero(typeof(d))
Base.zero(::Type{ADval{T}}) where {T}   = ADval{T}(zero(T), nothing)

Base.one(d::ADval{T}) where {T}         = one(typeof(d))
Base.one(::Type{ADval{T}}) where {T}    = ADval{T}(one(T), nothing)

Base.sign(d::ADval{T}) where {T} = sign(d.val)

###################
# Pretty printing #
###################

"compact form"
function Base.show(io::IO, d::ADval)
    print(io, typeof(d), "(", d.val, ", <derivnode>)")
end

"multiline form"
function Base.show(io::IO, ::MIME"text/plain", d::ADval)
    println(io, typeof(d))
    println(io, "  val=", d.val)
    println(io, "  deriv:\n", deriv(d))  
end

###################
# Get derivatives #
###################

"evaluate derivnode by traversing tree, and return as a sparse vector"
function deriv(d::ADval{T}, N) where {T}
    I = Vector{Int64}()
    V = Vector{T}()
    reduce_deriv!(I, V, 1.0, d.derivnode)
    # reduce_deriv_tree!(I, V, 1.0, d.derivnode)
    return sparsevec(I, V, N)
end

function deriv(d::ADval{T}) where {T}
    I = Vector{Int64}()
    V = Vector{T}()
    reduce_deriv!(I, V, 1.0, d.derivnode)
    # reduce_deriv_tree!(I, V, 1.0, d.derivnode)
    return sparsevec(I, V)
end

"""
    jacobian(advec::Vector{ADval}, N) -> SparseMatrixCSC

Aggregate derivatives to form sparse Jacobian.
"""
function jacobian(advec::Vector{ADval{T}}, N) where {T}

    Nvar = length(advec)

    # @info "SparsityTracing.jacobian ($Nvar, $N)"
    
    # define arrays for sparse matrix creation
    I = Vector{Int64}()
    J = Vector{Int64}()
    V = Vector{T}()

    for i = 1:Nvar
        # @info "  i=$i"
        derivJ, derivV = findnz(deriv(advec[i], N))
        # @info "  length(derivJ) = $(length(derivJ))"
        for j = 1:length(derivJ)
            push!(I, i)
            push!(J, derivJ[j])
            push!(V, derivV[j])
        end
    end 

    return sparse(I, J, V, Nvar, N)
end



"doesn't need to be mutable, but slightly faster (with Julia 1.5)"
mutable struct DerivLeaf <: AbstractDerivNode
    index::Int64
end

"doesn't need to be mutable, but slightly faster (with Julia 1.5)"
mutable struct DerivNode{T} <: AbstractDerivNode
    lmult::T
    lnode::Union{Nothing, AbstractDerivNode}
    rmult::T
    rnode::Union{Nothing, AbstractDerivNode}
end

function reduce_deriv!(I, v, mult, node::Nothing)
    return nothing
end

function reduce_deriv!(I, V, mult, node::DerivLeaf)
    push!(I, node.index)
    push!(V, mult)
    return nothing
end

function reduce_deriv!(I, V, mult, node::DerivNode)
    reduce_deriv!(I, V, mult*node.lmult, node.lnode)
    reduce_deriv!(I, V, mult*node.rmult, node.rnode)
    return nothing
end

function reduce_deriv_tree!(I, V, mult, root::Union{Nothing, DerivNode, DerivLeaf})

    stack = Vector{Tuple{Float64, Union{Nothing, DerivNode, DerivLeaf}}}()

    curr = (mult, root)
    npush = 0

    while curr[2] isa DerivNode || !isempty(stack)
        # reach the left-most node of the curr node
        while curr[2] isa DerivNode
            # put curr on stack and traverse left subtree            
            push!(stack, curr)
            mult, node = curr
            curr = (mult*node.lmult, node.lnode)
            npush += 1
            if npush % 1000000 == 0
                @info "    npush $npush length(stack, I, V) $(length(stack)) $(length(I)) $(length(V))"
            end
        end
        # curr[2] must be either Nothing or a DerivLeaf
        if curr[2] isa DerivLeaf
            mult, node = curr
            push!(I, node.index)
            push!(V, mult)
            # @info "    $(node.index)   $mult"
        end
        mult, node = pop!(stack)
        # we have visited the node and its left subtree
        # visit the right subtree
        curr = (mult*node.rmult, node.rnode)
    end

    # curr[2] must be either Nothing or a DerivLeaf
    if curr[2] isa DerivLeaf
        mult, node = curr
        push!(I, node.index)
        push!(V, mult)
        # @info "    $(node.index)   $mult"
    end


    # @info " reduce_deriv_tree! length(I) = $(length(I))"
    return nothing

end


##########################################
# Autogenerate functions using DiffRules #
##########################################

"display DiffRules rules (diagnostic for development)"
function list_diffrules(arity)
    for key in DiffRules.diffrules()
        if key[3] in arity
            println(key)
        end
    end
end

"generate a unary AD rule
   ns       eg :Base
   op       eg :sin"
function gen_unary_rule(ns, op)

    deriv = DiffRules.diffrule(ns, op, :(u.val)) # eg :(cos(u.val))

    @eval $ns.$op(u::ADval{T}) where {T} = ADval{T}($ns.$op(u.val), DerivNode(convert(T, $deriv), u.derivnode, zero(T), nothing))
end

"generate a binary AD rule
   ns       eg :Base
   op       eg :+"
function gen_binary_rule(ns, op)

    u_deriv, v_deriv = DiffRules.diffrule(ns, op, :(u.val), :(v.val)) 
    @eval $ns.$op(u::ADval{T}, v::ADval{T}) where {T} = ADval{T}($ns.$op(u.val, v.val), 
        DerivNode(convert(T, $u_deriv), u.derivnode, convert(T, $v_deriv), v.derivnode))

    u_deriv, dummy = DiffRules.diffrule(ns, op, :(u.val), :v) 
    @eval $ns.$op(u::ADval{T}, v::Real) where {T} = ADval{T}($ns.$op(u.val, v),
        DerivNode(convert(T, $u_deriv), u.derivnode, zero(T), nothing))

    dummy, v_deriv = DiffRules.diffrule(ns, op, :u, :(v.val)) 
    @eval $ns.$op(u::Real, v::ADval{T}) where {T} = ADval{T}($ns.$op(u, v.val),
        DerivNode(convert(T, $v_deriv), v.derivnode, zero(T), nothing))
end

"generate a logical rule (ignore derivatives, just test value)"
function gen_logical_rule(op)
    @eval Base.$op(u::ADval{T}, v::ADval{T}) where {T} = $op(u.val, v.val)
    @eval Base.$op(u::ADval{T}, v::Real) where {T} = $op(u.val, v)
    @eval Base.$op(u::Real, v::ADval{T}) where {T} = $op(u, v.val)
end

function gen_unary_rules(namespaces, excludeops)
    for (ns, op, arity) in DiffRules.diffrules()
        if ns in namespaces && arity == 1 && !(op in excludeops)
            gen_unary_rule(ns, op)
        end
    end
end

function gen_binary_rules(namespaces, excludeops)
    for (ns, op, arity) in DiffRules.diffrules()
        if ns in namespaces && arity == 2 && !(op in excludeops)
            gen_binary_rule(ns, op)
        end
    end
end

function gen_logical_rules()
    for op in (:<, :>, :>=, :<=, :(==))
        gen_logical_rule(op)
    end
end

# add functions to Base
gen_unary_rules((:Base, :SpecialFunctions), ())
gen_binary_rules((:Base, :SpecialFunctions), ())
gen_logical_rules()

###############
# TODO fixups #
###############
# Errors Base.:^(d::ADval{T}, n::Integer) is ambiguous ? 

Base.:^(d::ADval{T}, p::Integer) where {T} = d^Float64(p)

end