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
        v0 = res_odd ? big(0.0) : big(taylor[1])
        return quote
            if x == 0
                return $v0
            else
                return $e
            end
        end
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

macro _combined_func(name, threshold)
    quote
        @inline $(esc(name))(x::T, s, c) where T =
            fastabs(x) <= float(T)($threshold) ?
            $(esc(Symbol("_$(name)_small")))(float(x)) :
            $(esc(Symbol("_$(name)_big")))(float(x), s, c)
    end
end

# sin(x) / x
@inline _sin_c1_small(x::T) where T =
    @inline evalpoly(x^2, (T(1), -1 / T(6), 1 / T(120), -1 / T(5_040),
                           1 / T(362_880), -1 / T(39_916_800), 1 / T(6_227_020_800)))
@inline _sin_c1_big(x, s, c) = s / x
@_combined_func sin_c1 0.3

# (sin(x) - x * cos(x)) / x^2
@inline _sin_c2_small(x::T) where T =
    x * @inline evalpoly(x^2, (1 / T(3), -1 / T(30), 1 / T(840), -1 / T(45_360),
                               1 / T(3_991_680), -1 / T(518_918_400),
                               1 / T(93_405_312_000)))
@inline _sin_c2_big(x, s, c) = (s - c * x) / x^2
@_combined_func sin_c2 0.5

# ((x^2 - 2) * sin(x) + 2 * x * cos(x)) / x^3
@inline _sin_c3_small(x::T) where T =
    @inline evalpoly(x^2, (1 / T(3), -1 / T(10), 1 / T(168), -1 / T(6_480),
                           1 / T(443_520), -1 / T(47_174_400),
                           1 / T(7_185_024_000), -1 / T(1_482_030_950_400)))
@inline _sin_c3_big(x, s, c) = 2 * (x * c - s) / x^3 + s / x
@_combined_func sin_c3 0.7

# (1 - cos(x)) / x^2
@inline _cos_f1_small(x::T) where T =
    @inline evalpoly(x^2, (1 / T(2), -1 / T(24), 1 / T(720), -1 / T(40_320),
                           1 / T(3_628_800), -1 / T(479_001_600),
                           1 / T(87_178_291_200)))
@inline _cos_f1_big(x, s, c) = (1 - c) / x^2
@_combined_func cos_f1 0.45

# (x - sin(x)) / x^2
@inline _sin_f1_small(x::T) where T =
    x * @inline evalpoly(x^2, (1 / T(6), -1 / T(120), 1 / T(5_040), -1 / T(362_880),
                               1 / T(39_916_800), -1 / T(6_227_020_800)))
@inline _sin_f1_big(x, s, c) = (x - s) / x^2
@_combined_func sin_f1 0.3

# (2 - 2 * cos(x) - x * sin(x)) / x^3
@inline _cos_f2_small(x::T) where T =
    x * @inline evalpoly(x^2, (1 / T(12), -1 / T(180), 1 / T(6_720), -1 / T(453_600),
                               1 / T(47_900_160), -1 / T(7_264_857_600),
                               1 / T(1_494_484_992_000)))
@inline _cos_f2_big(x, s, c) = (2 * (1 - c) - s * x) / x^3
@_combined_func cos_f2 0.55

# (2 * sin(x) - x * cos(x) - x) / x^3
@inline _sin_f2_small(x::T) where T =
    @inline evalpoly(x^2, (1 / T(6), -1 / T(40), 1 / T(1_008), -1 / T(51_840),
                           1 / T(4_435_200), -1 / T(566_092_800),
                           1 / T(100_590_336_000), -1 / T(23_712_495_206_400)))
@inline _sin_f2_big(x, s, c) = ((s - x) + (s - x * c)) / x^3
@_combined_func sin_f2 0.7

# (x^2/2 + 1 - cos(x) - x * sin(x)) / x^4
@inline _cos_f3_small(x::T) where T =
    @inline evalpoly(x^2, (1 / T(8), -1 / T(144), 1 / T(5_760), -1 / T(403_200),
                           1 / T(43_545_600), -1 / T(6_706_022_400),
                           1 / T(1_394_852_659_200)))
@inline _cos_f3_big(x, s, c) = (x * (x / 2 - s) + (1 - c)) / (x^2)^2
@_combined_func cos_f3 0.6

# (x^3/3 - sin(x) + x * cos(x)) / x^4
@inline _sin_f3_small(x::T) where T =
    x * @inline evalpoly(x^2, (1 / T(30), -1 / T(840), 1 / T(45_360),
                               -1 / T(3_991_680), 1 / T(518_918_400),
                               -1 / T(93_405_312_000), 1 / T(22_230_464_256_000)))
@inline _sin_f3_big(x, s, c) = (x^3 / 3 + (x * c - s)) / (x^2)^2
@_combined_func sin_f3 0.7

# (-x^3/3 + (4 - x^2) * sin(x) - 4 * x * cos(x)) / x^5
@inline _sin_f4_small(x::T) where T =
    @inline evalpoly(x^2, (1 / T(30), -1 / T(280), 1 / T(9_072),
                           -1 / T(570_240), 1 / T(57_657_600),
                           -1 / T(8_491_392_000), 1 / T(1_710_035_712_000),
                           -1 / T(450_537_408_921_600)))
@inline _sin_f4_big(x, s, c) = (-x^2 * s + 4 * (s - x * c) - x^3 / 3) / (x^2) / (x^3)
@_combined_func sin_f4 1.1

# (-2/3 x^3 + (20 - 7x^2) * sin(x) + (x^2 - 20) * x * cos(x)) / x^6
@inline _sin_f5_small(x::T) where T =
    x * @inline evalpoly(x^2, (1 / T(140), -1 / T(2_268), 1 / T(95_040),
                               -1 / T(7_207_200), 1 / T(849_139_200),
                               -1 / T(142_502_976_000), 1 / T(32_181_243_494_400),
                               -1 / T(9_391_717_310_976_000)))
@inline _sin_f5_big(x::T, s, c) where T =
    (x^3 * (c - T(2) / 3) + (20 * (s - x * c) - 7 * x^2 * s)) / (x^3)^2
@_combined_func sin_f5 1.3

end
