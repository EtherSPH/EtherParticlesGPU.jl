#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/22 01:14:34
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

# keys(x::NamedTuple) will return the Tuple symbols
# values(x::NamedTuple) will return the Tuple values

@inline function keysVector(x::NamedTuple)::Vector{Symbol}
    list = []
    for item in keys(x)
        push!(list, item)
    end
    return list .|> Symbol
end

@inline function valuesVector(x::NamedTuple)
    list = []
    for item in values(x)
        @assert isa(item, Integer)
        push!(list, item)
    end
    return list
end

abstract type AbstractNamedIndexTable{IT <: Integer} end

@inline function get_n_capacity(x::AbstractNamedIndexTable{IT})::IT where {IT <: Integer}
    return x.n_capacity_
end

@inline function get_index_named_tuple(x::AbstractNamedIndexTable{IT})::NamedTuple where {IT <: Integer}
    return x.index_named_tuple_
end

struct NamedIndexTable{IT <: Integer} <: AbstractNamedIndexTable{IT}
    capacity_named_tuple_::NamedTuple
    index_named_tuple_::NamedTuple
    n_capacity_::IT
    name_symbol_list_::Vector{Symbol}
    name_string_list_::Vector{String}
    name_symbol_table_head_::Vector{Symbol}
    name_string_table_head_::Vector{String}
end

@inline function NamedIndexTable{IT}(capacity_named_tuple::NamedTuple)::NamedIndexTable{IT} where {IT <: Integer}
    @assert length(capacity_named_tuple) > 0
    name_symbol_list = keysVector(capacity_named_tuple)
    name_string_list = String.(name_symbol_list)
    capacity_values_list = IT.(valuesVector(capacity_named_tuple))
    n_capacity::IT = sum(capacity_values_list)
    index_values_list = zeros(IT, length(capacity_values_list))
    index_values_list[1] = IT(1)
    for i in 2:length(capacity_values_list)
        strat = index_values_list[i - 1]
        index_values_list[i] = strat + capacity_values_list[i - 1]
    end
    index_named_tuple = NamedTuple{Tuple(name_symbol_list)}(index_values_list)
    capacity_named_tuple = NamedTuple{Tuple(name_symbol_list)}(capacity_values_list)
    name_symbol_table_head = Vector{Symbol}(undef, n_capacity)
    index::IT = 1
    for i in eachindex(name_symbol_list)
        if capacity_named_tuple[i] == 1
            name_symbol_table_head[index] = name_symbol_list[i]
            index += 1
        else
            for j in 1:capacity_named_tuple[i]
                name_symbol_table_head[index] = Symbol(string(name_symbol_list[i], "_", j))
                index += 1
            end
        end
    end
    name_string_table_head = String.(name_symbol_table_head)
    return NamedIndexTable{IT}(
        capacity_named_tuple,
        index_named_tuple,
        n_capacity,
        name_symbol_list,
        name_string_list,
        name_symbol_table_head,
        name_string_table_head,
    )
end
