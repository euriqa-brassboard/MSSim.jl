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

_sin_taylor_coeff(n) = [(-1)^k // factorial(big(k) * 2 + 1) for k in 0:(n - 1)]
_cos_taylor_coeff(n) = [(-1)^k // factorial(big(k) * 2) for k in 0:(n - 1)]

struct _Term
    has_sin::Bool
    has_cos::Bool
    order::Int
    coeff::Real
end

struct _Exp
    left::Union{_Exp,_Term}
    right::Union{_Exp,_Term}
end

function _mul_exp(@nospecialize(e1), @nospecialize(e2))
    if e1 == 1
        return e2
    elseif e2 == 1
        return e1
    elseif e1 == -1
        return :(-$e2)
    elseif e2 == -1
        return :(-$e1)
    else
        return :($e1 * $e2)
    end
end

function _xpower_sym(power)
    if power == 1
        return :x
    else
        return Symbol("x$power")
    end
end

function __convert_exp(@nospecialize(exp), T, powers)
    if isa(exp, _Exp)
        t1, e1 = __convert_exp(exp.left, T, powers)
        t2, e2 = __convert_exp(exp.right, T, powers)
        t = _Term(t1.has_sin && t2.has_sin,
                  t1.has_cos && t2.has_cos,
                  min(t1.order, t2.order), 1)

        if t1.has_sin && !t.has_sin
            e1 = _mul_exp(e1, :s)
        end
        if t1.has_cos && !t.has_cos
            e1 = _mul_exp(e1, :c)
        end
        xorder1 = t1.order - t.order
        if xorder1 > 1
            push!(powers, xorder1)
            e1 = _mul_exp(e1, _xpower_sym(xorder1))
        elseif xorder1 == 1
            e1 = _mul_exp(e1, :x)
        end
        if t1.coeff != 1
            e1 = _mul_exp(e1, T(t1.coeff))
        end

        if t2.has_sin && !t.has_sin
            e2 = _mul_exp(e2, :s)
        end
        if t2.has_cos && !t.has_cos
            e2 = _mul_exp(e2, :c)
        end
        xorder2 = t2.order - t.order
        if xorder2 > 1
            push!(powers, xorder2)
            e2 = _mul_exp(e2, _xpower_sym(xorder2))
        elseif xorder2 == 1
            e2 = _mul_exp(e2, :x)
        end
        if t2.coeff != 1
            e2 = _mul_exp(e2, T(t2.coeff))
        end

        return t, :($e1 + $e2)
    end
    return exp::_Term, 1
end

function _convert_exp(@nospecialize(exp), T, powers)
    t, e = __convert_exp(exp, T, powers)
    if t.has_sin
        e = _mul_exp(e, :s)
    end
    if t.has_cos
        e = _mul_exp(e, :c)
    end
    if t.order > 1
        push!(powers, t.order)
        e = _mul_exp(e, _xpower_sym(t.order))
    elseif t.order == 1
        e = _mul_exp(e, :x)
    end
    if t.coeff != 1
        e = _mul_exp(e, T(t.coeff))
    end
    return e
end

function _compute_power(power, powers, computed, exp)
    if power in computed
        return
    end
    p1 = power ÷ 2
    p2 = power - p1
    _compute_power(p1, powers, computed, exp)
    if p2 != p1
        _compute_power(p2, powers, computed, exp)
    end
    p1sym = _xpower_sym(p1)
    p2sym = _xpower_sym(p2)
    push!(exp.args, :($(_xpower_sym(power)) = $p1sym * $p2sym))
    push!(computed, power)
    delete!(powers, power)
    return
end

function _compute_powers(powers)
    max_power = maximum(powers)
    exp = quote end
    pwr = 2
    computed = Set([1])
    while !isempty(powers)
        _compute_power(pop!(powers), powers, computed, exp)
    end
    return exp
end

function _num_terms(x, odd)
    nterms = 1
    # Approximate the accuracy of the taylor expansion with x^n/n!
    accuracy = odd ? x : 1.0
    while accuracy > 1e-15
        accuracy *= x^2 / (2 * nterms + odd - 1) / (2 * nterms + odd)
        nterms += 1
    end
    return nterms
end

function _get_full_terms(v, order_idx, base)
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

function _pop_leading_terms(all_terms)
    if length(all_terms) == 1
        term = all_terms[1].second
        if popfirst!(term) != 0
            error("Diverging sequence.")
        end
        return all_terms
    end
    new_terms = Pair{Union{_Exp,_Term},Vector{Real}}[]
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
                    term = _Exp(term, te)
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

function _merge_terms(all_terms)
    if length(all_terms) == 1
        return all_terms[1]
    end
    mid = length(all_terms) ÷ 2
    left = _merge_terms(@view all_terms[1:mid])
    right = _merge_terms(@view all_terms[mid + 1:end])
    return Pair(_Exp(left.first, right.first), left.second .+ right.second)
end

function _plan_trig_ratio(T, odd_num, div_pow, plain_poly, sin_poly, cos_poly)
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
    res_terms = _num_terms(threshold, res_odd)
    # We'll expand to up to `2 * res_terms - 1`-th order in the final result
    # (`res_terms` terms in total)
    # This is the number of terms we need to compute in the numerator
    # including the lowest terms
    max_terms = (div_pow + 2 * res_terms - 1 - odd_num) ÷ 2 + 1
    sin_taylor = _sin_taylor_coeff(max_terms)
    cos_taylor = _cos_taylor_coeff(max_terms)
    plain_taylor = [Int(i == 1) for i in 1:max_terms]

    all_terms = Pair{Union{_Exp,_Term},Vector{Real}}[]
    for (i, v) in enumerate(sin_poly)
        if v == 0
            continue
        end
        if odd_num
            push!(all_terms,
                  _Term(true, false, i * 2 - 2, v)=>_get_full_terms(v, i, sin_taylor))
        else
            push!(all_terms,
                  _Term(true, false, i * 2 - 1, v)=>_get_full_terms(v, i + 1,
                                                                    sin_taylor))
        end
    end
    for (i, v) in enumerate(cos_poly)
        if v == 0
            continue
        end
        order = odd_num ? i * 2 - 1 : i * 2 - 2
        push!(all_terms, _Term(false, true, order, v)=>_get_full_terms(v, i,
                                                                       cos_taylor))
    end
    for (i, v) in enumerate(plain_poly)
        if v == 0
            continue
        end
        order = odd_num ? i * 2 - 1 : i * 2 - 2
        push!(all_terms, _Term(false, false, order, v)=>_get_full_terms(v, i,
                                                                        plain_taylor))
    end
    for i in 1:((div_pow + 1 - odd_num) ÷ 2)
        all_terms = _pop_leading_terms(all_terms)
    end
    term, taylor = _merge_terms(all_terms)
    powers = Set{Int}()
    e = _convert_exp(term, T, powers)
    if div_pow > 1
        e = :($e / $(Symbol("x$div_pow")))
        push!(powers, div_pow)
    else
        e = :($e / x)
    end
    if !isempty(powers)
        e = :($(_compute_powers(powers)); $e)
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

function _gen_poly_trig_ratio(T, odd, taylor)
    poly_expr = :(evalpoly(x * x, ($(T.(taylor)...),)))
    if odd
        poly_expr = :(x * $poly_expr)
    end
    return poly_expr
end

struct TrigRatio{odd,div,plain_poly,sin_poly,cos_poly} <: Function
end

@generated function (::TrigRatio{odd,div,plain_poly,sin_poly,cos_poly})(
    x::T, s, c) where {T,odd,div,plain_poly,sin_poly,cos_poly}

    T = float(T)
    res_odd, threshold, e, taylor =
        _plan_trig_ratio(T, odd, div, plain_poly, sin_poly, cos_poly)
    if T === BigFloat
        # Still use the taylor expansion expression
        # to support a few orders of automatic differentiation
        # but use the normal expression for everywhere other than 0
        threshold = 0
    end
    poly_expr = _gen_poly_trig_ratio(T, res_odd, taylor)
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
# ((x^2 - 2) * sin(x) + 2 * x * cos(x)) / x^3
const sin_c3 = TrigRatio{true,3,(),(-2,1),(2,)}()
# (1 - cos(x)) / x^2
const cos_f1 = TrigRatio{false,2,(1,),(),(-1,)}()
# (x - sin(x)) / x^2
const sin_f1 = TrigRatio{true,2,(1,),(-1,),()}()
# (2 - 2 * cos(x) - x * sin(x)) / x^3
const cos_f2 = TrigRatio{false,3,(2,),(-1,),(-2,)}()
# (2 * sin(x) - x * cos(x) - x) / x^3
const sin_f2 = TrigRatio{true,3,(-1,),(2,),(-1,)}()
# (x^2/2 + 1 - cos(x) - x * sin(x)) / x^4
const cos_f3 = TrigRatio{false,4,(1,1//2),(-1,),(-1,)}()
# (x^3/3 - sin(x) + x * cos(x)) / x^4
const sin_f3 = TrigRatio{true,4,(0,1//3),(-1,),(1,)}()
# (-x^3/3 + (4 - x^2) * sin(x) - 4 * x * cos(x)) / x^5
const sin_f4 = TrigRatio{true,5,(0,-1//3),(4,-1),(-4,)}()
# (-2/3 x^3 + (20 - 7x^2) * sin(x) + (x^2 - 20) * x * cos(x)) / x^6
const sin_f5 = TrigRatio{true,6,(0,-2//3),(20,-7),(-20,1)}()

end
