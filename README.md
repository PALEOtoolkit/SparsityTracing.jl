# ADelemtree.jl

Automatic Jacobian sparsity detection using minimal scalar tracing and autodifferentiation.

## Installation

The package is not yet registered, so add it using:

```julia
julia> ] add https://github.com/sjdaines/ADelemtree.jl
```

## Example
```julia
julia> import ADelemtree as AD

julia> function rober(du,u,p)
                   y₁,y₂,y₃ = u
                   k₁,k₂,k₃ = p
                   du[1] = -k₁*y₁+k₃*y₂*y₃
                   du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
                   du[3] =  k₂*y₂^2
                   nothing
               end;

julia> u = [1.0, 2.0, 3.0];

julia> p = (0.04,3e7,1e4);

julia> u_ad = AD.create_advec(u);

julia> du_ad = similar(u_ad);

julia> rober(du_ad, u_ad, p)

julia> AD.deriv(du_ad[3])
2-element SparseArrays.SparseVector{Float64, Int64} with 1 stored entry:
  [2]  =  1.2e8

julia> Jad = AD.jacobian(du_ad, length(du_ad))
3×3 SparseArrays.SparseMatrixCSC{Float64, Int64} with 7 stored entries:
 -0.04  30000.0        20000.0
  0.04     -1.2003e8  -20000.0
   ⋅        1.2e8           ⋅ 

```

## Implementation

Implements a scalar type `ADelemtree.ADval{T<:Real}` that holds a value and a binary tree of scalar derivatives. The tree is initialised to a Vector of leaf nodes by `ADelemtree.create_advec`.  It is populated with derivatives calculated by `DiffRules` when a Julia function is called (eg a function `y = f(x)` calculating the RHS of an ODE). `ADelemtree.jacobian` then walks the tree and calculates the Jacobian as a sparse matrix.  This provides a robust way of detecting Jacobian sparsity (and for test purposes only, a very slow way of calculating the actual derivative).  The sparsity pattern may then be used to generate matrix colouring for a fast AD package eg `SparseDiffTools`.
