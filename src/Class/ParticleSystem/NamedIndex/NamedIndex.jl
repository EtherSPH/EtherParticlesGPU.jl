#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/24 16:41:26
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

include("NamedIndexTable.jl")

abstract type AbstractNamedIndex{IT <: Integer} end

struct NamedIndex{IT <: Integer} <: AbstractNamedIndex{IT}
    int_named_index_table_::NamedIndexTable{IT}
    float_named_index_table_::NamedIndexTable{IT}
    combined_index_named_tuple_::NamedTuple # combine int and float index_named_tuple_
end

@inline function NamedIndex{IT}(
    int_named_tuple::NamedTuple,
    float_named_tuple::NamedTuple,
)::NamedIndex{IT} where {IT <: Integer}
    int_named_index_table = NamedIndexTable{IT}(int_named_tuple)
    float_named_index_table = NamedIndexTable{IT}(float_named_tuple)
    combined_index_named_tuple =
        merge(get_index_named_tuple(int_named_index_table), get_index_named_tuple(float_named_index_table))
    return NamedIndex{IT}(int_named_index_table, float_named_index_table, combined_index_named_tuple)
end

@inline function get_index_named_tuple(named_index::NamedIndex{IT})::NamedTuple where {IT <: Integer}
    return named_index.combined_index_named_tuple_
end

@inline function get_int_n_capacity(named_index::NamedIndex{IT})::IT where {IT <: Integer}
    return get_n_capacity(named_index.int_named_index_table_)
end

@inline function get_float_n_capacity(named_index::NamedIndex{IT})::IT where {IT <: Integer}
    return get_n_capacity(named_index.float_named_index_table_)
end
