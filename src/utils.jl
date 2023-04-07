#!/usr/bin/julia

module Utils

struct JaggedMatrix{T} <: AbstractVector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}
    values::Vector{T}
    idx_ranges::Vector{UnitRange{Int}}
    function JaggedMatrix{T}() where T
        return new(T[], UnitRange{Int}[])
    end
    global _similar
    @inline function _similar(m::JaggedMatrix, ::Type{T}) where T
        return new{T}(similar(m.values, T), copy(m.idx_ranges))
    end
end

Base.size(m::JaggedMatrix) = size(m.idx_ranges)
Base.length(m::JaggedMatrix) = length(m.idx_ranges)

Base.@propagate_inbounds function Base.getindex(m::JaggedMatrix, idx)
    rng = m.idx_ranges[idx]
    return @view m.values[rng]
end

Base.similar(m::JaggedMatrix{T}) where T = _similar(m, T)
Base.similar(m::JaggedMatrix, ::Type{AT}) where AT <: AbstractVector{T} where T =
    _similar(m, T)

function Base.push!(m::JaggedMatrix{T}, ary::AbstractArray) where T
    last_idx = length(m.values)
    neles = length(ary)
    append!(m.values, ary)
    push!(m.idx_ranges, (last_idx + 1):(last_idx + neles))
    return m
end

function Base.push!(m::JaggedMatrix{T}, ele::T) where T
    rng = m.idx_ranges[end]
    new_rng = first(rng):(last(rng) + 1)
    push!(m.values, ele)
    m.idx_ranges[end] = new_rng
    return m
end

Base.push!(m::JaggedMatrix{T}, ele) where T = push!(m, convert(T, ele))

function Base.empty!(m::JaggedMatrix)
    empty!(m.values)
    empty!(m.idx_ranges)
    return m
end

# Math utility functions

@inline mulim(x) = @inline complex(-imag(x), real(x))

fastabs(x::Number) = abs(x)
fastabs(z::Complex) = abs(real(z)) + abs(imag(z))

module TR

# First n Taylor coefficients for sin and cos
sin_taylor_coeffs(n) = [(-1)^k // factorial(big(k) * 2 + 1) for k in 0:(n - 1)]
cos_taylor_coeffs(n) = [(-1)^k // factorial(big(k) * 2) for k in 0:(n - 1)]

struct Term
    has_sin::Bool
    has_cos::Bool
    order::Int
    coeff::Real
end

struct Exp
    left::Union{Exp,Term}
    right::Union{Exp,Term}
end

# Multiply julia expressions
function mul_expr(@nospecialize(e1), @nospecialize(e2))
    if e1 == 1
        return e2
    elseif e2 == 1
        return e1
    elseif isa(e1, Number) && isinteger(e1)
        return :($(Int(e1)) * $e2)
    elseif isa(e2, Number) && isinteger(e2)
        return :($(Int(e2)) * $e1)
    else
        return :($e1 * $e2)
    end
end

function add_expr(@nospecialize(e1), @nospecialize(e2))
    if e1 == 0
        return e2
    elseif e2 == 0
        return e1
    elseif Meta.isexpr(e1, :call) && length(e1.args) == 3 && e1.args[1] === :*
        return :(muladd($(e1.args[2]), $(e1.args[3]), $e2))
    elseif Meta.isexpr(e2, :call) && length(e2.args) == 3 && e2.args[1] === :*
        return :(muladd($(e2.args[2]), $(e2.args[3]), $e1))
    else
        return :($e1 + $e2)
    end
end

# Symbol for power of x
function xpower_sym(power)
    if power == 1
        return :x
    else
        return Symbol("x$power")
    end
end

# Convert a Term to a julia expression
function gen_term(base, T, has_sin, has_cos, xorder, coeff, powers)
    e = base
    if has_sin
        e = mul_expr(e, :s)
    end
    if has_cos
        e = mul_expr(e, :c)
    end
    if xorder > 1
        push!(powers, xorder)
        e = mul_expr(e, xpower_sym(xorder))
    elseif xorder == 1
        e = mul_expr(e, :x)
    end
    if coeff != 1
        e = mul_expr(e, T(coeff))
    end
    return e
end

# Convert a Exp to a julia expression
function _gen_exp(@nospecialize(exp), T, powers)
    if isa(exp, Exp)
        t1, e1 = _gen_exp(exp.left, T, powers)
        t2, e2 = _gen_exp(exp.right, T, powers)
        t = Term(t1.has_sin && t2.has_sin,
                 t1.has_cos && t2.has_cos,
                 min(t1.order, t2.order), 1)
        e1 = gen_term(e1, T, t1.has_sin && !t.has_sin, t1.has_cos && !t.has_cos,
                      t1.order - t.order, t1.coeff, powers)
        e2 = gen_term(e2, T, t2.has_sin && !t.has_sin, t2.has_cos && !t.has_cos,
                      t2.order - t.order, t2.coeff, powers)
        return t, add_expr(e1, e2)
    end
    return exp::Term, 1
end

function gen_exp(@nospecialize(exp), T, powers)
    t, e = _gen_exp(exp, T, powers)
    return gen_term(e, T, t.has_sin, t.has_cos, t.order, t.coeff, powers)
end

# Make sure the x powers required are all computed
function compute_xpower(power, powers, computed, exp)
    if power in computed
        return
    end
    p1 = power ÷ 2
    p2 = power - p1
    compute_xpower(p1, powers, computed, exp)
    if p2 != p1
        compute_xpower(p2, powers, computed, exp)
    end
    p1sym = xpower_sym(p1)
    p2sym = xpower_sym(p2)
    push!(exp.args, :($(xpower_sym(power)) = $p1sym * $p2sym))
    push!(computed, power)
    delete!(powers, power)
    return
end

function compute_xpowers(powers)
    max_power = maximum(powers)
    exp = quote end
    pwr = 2
    computed = Set([1])
    while !isempty(powers)
        compute_xpower(pop!(powers), powers, computed, exp)
    end
    return exp
end

function estimate_num_terms(T, x, odd)
    if T === BigFloat
        # We only use this to provide support for automatic differentiation
        return 11
    end
    nterms = 1
    # Approximate the accuracy of the taylor expansion with x^n/n!
    accuracy = odd ? x : 1.0
    while accuracy > 2 * eps(T)
        accuracy *= x^2 / (2 * nterms + odd - 1) / (2 * nterms + odd)
        nterms += 1
    end
    return nterms
end

function get_full_taylor(v, order_idx, base)
    terms = base .* v
    if order_idx == 1
        return terms
    end
    nterms = length(terms)
    for i in (nterms - order_idx + 1):-1:1
        terms[i + order_idx - 1] = terms[i]
    end
    for i in 1:(order_idx - 1)
        terms[i] = 0
    end
    return terms
end

function pop_leading_taylor(all_terms)
    if length(all_terms) == 1
        term = all_terms[1].second
        if popfirst!(term) != 0
            error("Diverging sequence.")
        end
        return all_terms
    end
    new_terms = Pair{Union{Exp,Term},Vector{Real}}[]
    non_zero_terms = Pair{Real,Int}[]
    for (i, (term, taylor)) in enumerate(all_terms)
        v0 = popfirst!(taylor)
        if v0 == 0
            push!(new_terms, term=>taylor)
        else
            push!(non_zero_terms, v0=>i)
        end
    end
    if !isempty(non_zero_terms)
        sort!(non_zero_terms, by=x->abs(x.first), rev=true)
        cur_value, i = popfirst!(non_zero_terms)
        term, taylor = all_terms[i]
        while !isempty(non_zero_terms)
            found = false
            for (i, (v, term_idx)) in enumerate(non_zero_terms)
                if v * cur_value <= 0
                    cur_value += v
                    te, ta = all_terms[term_idx]
                    term = Exp(term, te)
                    taylor = taylor .+ ta
                    found = true
                    popat!(non_zero_terms, i)
                    break
                end
            end
            if !found
                error("Diverging sequence.")
            end
        end
        if cur_value != 0
            error("Diverging sequence.")
        end
        push!(new_terms, term=>taylor)
    end
    return new_terms
end

function merge_terms(all_terms)
    if length(all_terms) == 1
        return all_terms[1]
    end
    mid = length(all_terms) ÷ 2
    left = merge_terms(@view all_terms[1:mid])
    right = merge_terms(@view all_terms[mid + 1:end])
    return Pair(Exp(left.first, right.first), left.second .+ right.second)
end

function plan(T, odd_num, div_pow, plain_poly, sin_poly, cos_poly)
    # * odd_num == true:
    #   plain_poly: x + x^3 ... lowest order: x
    #   sin_poly: 1 + x^2 ... lowest order: x
    #   cos_poly: x + x^3 ... lowest order: x
    # * odd_num == false:
    #   plain_poly: 1 + x^2 ... lowest order: 1
    #   sin_poly: x + x^3 ... lowest order: x^2
    #   cos_poly: 1 + x^2 ... lowest order: 1
    res_odd = odd_num ⊻ isodd(div_pow)
    # Now the lowest order for the three terms are the same (1 for even and x for odd)
    nterms = (count(!=(0), plain_poly) + count(!=(0), sin_poly) +
        count(!=(0), cos_poly))
    # A very rough estimate on the error (for Float64)
    threshold = nterms^(1 / div_pow)
    res_terms = estimate_num_terms(T, threshold, res_odd)
    # We'll expand to up to `2 * res_terms - 1`-th order in the final result
    # (`res_terms` terms in total)
    # This is the number of terms we need to compute in the numerator
    # including the lowest terms
    max_terms = (div_pow + 2 * res_terms - 1 - odd_num) ÷ 2 + 1
    sin_taylor = sin_taylor_coeffs(max_terms)
    cos_taylor = cos_taylor_coeffs(max_terms)
    plain_taylor = [Int(i == 1) for i in 1:max_terms]

    all_terms = Pair{Union{Exp,Term},Vector{Real}}[]
    for (i, v) in enumerate(sin_poly)
        if v == 0
            continue
        end
        if odd_num
            push!(all_terms,
                  Term(true, false, i * 2 - 2, v)=>get_full_taylor(v, i, sin_taylor))
        else
            push!(all_terms,
                  Term(true, false, i * 2 - 1, v)=>get_full_taylor(v, i + 1,
                                                                   sin_taylor))
        end
    end
    for (i, v) in enumerate(cos_poly)
        if v == 0
            continue
        end
        order = odd_num ? i * 2 - 1 : i * 2 - 2
        push!(all_terms, Term(false, true, order, v)=>get_full_taylor(v, i,
                                                                      cos_taylor))
    end
    for (i, v) in enumerate(plain_poly)
        if v == 0
            continue
        end
        order = odd_num ? i * 2 - 1 : i * 2 - 2
        push!(all_terms, Term(false, false, order, v)=>get_full_taylor(v, i,
                                                                       plain_taylor))
    end
    for i in 1:((div_pow + 1 - odd_num) ÷ 2)
        all_terms = pop_leading_taylor(all_terms)
    end
    term, taylor = merge_terms(all_terms)
    powers = Set{Int}()
    e = gen_exp(term, T, powers)
    if div_pow > 1
        e = :($e / $(Symbol("x$div_pow")))
        push!(powers, div_pow)
    else
        e = :($e / x)
    end
    if !isempty(powers)
        e = :($(compute_xpowers(powers)); $e)
    end
    for i in length(taylor):-1:1
        order = res_odd ? (i * 2 - 1) : (i * 2 - 2)
        if abs(threshold^order * taylor[i]) > eps(T)
            break
        end
        pop!(taylor)
    end
    return res_odd, threshold, e, taylor
end

function gen_poly_expr(T, odd, taylor)
    poly_expr = :(evalpoly(x * x, ($(T.(taylor)...),)))
    if odd
        poly_expr = :(x * $poly_expr)
    end
    return poly_expr
end
end

struct TrigRatio{odd,div,plain_poly,sin_poly,cos_poly} <: Function
end

@generated function (::TrigRatio{odd,div,plain_poly,sin_poly,cos_poly})(
    x::T, s, c) where {T,odd,div,plain_poly,sin_poly,cos_poly}

    T = float(T)
    res_odd, threshold, e, taylor = TR.plan(T, odd, div, plain_poly,
                                            sin_poly, cos_poly)
    if T === BigFloat
        # Still use the taylor expansion expression
        # to support a few orders of automatic differentiation
        # but use the normal expression for everywhere other than 0
        threshold = 0
    end
    poly_expr = TR.gen_poly_expr(T, res_odd, taylor)
    return quote
        @inline if fastabs(x) > $(T(threshold))
            return $e
        else
            return $poly_expr
        end
    end
end

# sin(x) / x
const sin_c1 = TrigRatio{true,1,(),(1,),()}()

# (sin(x) - x * cos(x)) / x^2
const sin_c2 = TrigRatio{true,2,(),(1,),(-1,)}()

# ((-2 + x^2) * sin(x) + 2 * x * cos(x)) / x^3
const sin_c3 = TrigRatio{true,3,(),(-2,1),(2,)}()

# (1 - cos(x)) / x^2
const cos_f1 = TrigRatio{false,2,(1,),(),(-1,)}()

# (x - sin(x)) / x^3
const sin_f1 = TrigRatio{true,3,(1,),(-1,),()}()

# (2 - x * sin(x) - 2 * cos(x)) / x^3
const cos_f2 = TrigRatio{false,3,(2,),(-1,),(-2,)}()

# (-x + 2 * sin(x) - x * cos(x)) / x^3
const sin_f2 = TrigRatio{true,3,(-1,),(2,),(-1,)}()

# (1 + x^2/2 - x * sin(x) - cos(x)) / x^4
const cos_f3 = TrigRatio{false,4,(1,1//2),(-1,),(-1,)}()

# (-6 + 4 * x * sin(x) + (6 - x^2) * cos(x)) / x^4
const cos_f3_2 = TrigRatio{false,4,(-6,),(4,),(6,-1)}()

# (1/3 * x^3 - sin(x) + x * cos(x)) / x^4
const sin_f3 = TrigRatio{true,4,(0,1//3),(-1,),(1,)}()

# (-2 * x + (6 - x^2) * sin(x) - 4 * x * cos(x)) / x^4
const sin_f3_2 = TrigRatio{true,4,(-2,),(6,-1),(-4,)}()

# ((-6 + 3x^2) * sin(x) + (6 * x - x^3) * cos(x)) / x^4
const sin_f3_3 = TrigRatio{true,4,(),(-6,3),(6,-1)}()

# (-1/3 * x^3 + (4 - x^2) * sin(x) - 4 * x * cos(x)) / x^5
const sin_f4 = TrigRatio{true,5,(0,-1//3),(4,-1),(-4,)}()

# (-2/3 * x^3 + (20 - 7x^2) * sin(x) + (-20 + x^2) * x * cos(x)) / x^6
const sin_f5 = TrigRatio{true,6,(0,-2//3),(20,-7),(-20,1)}()

end
