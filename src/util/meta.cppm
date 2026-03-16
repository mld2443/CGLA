module;

#include <cstddef>     // size_t
#include <type_traits> // conditional_t, remove_reference_t
#include <utility>     // forward, index_sequence, make_index_sequence

export module meta;


// Template Metaprogramming
export namespace meta {
    template <class, std::size_t... DIMS>
    struct ArrayShape {
        static constexpr std::size_t RANK = sizeof...(DIMS);
        static constexpr std::size_t VALUE[] = { DIMS... };
    };

    template<class T, std::size_t N, std::size_t... DIMS>
    struct ArrayShape<T[N], DIMS...> : ArrayShape<T, DIMS..., N> { using Base = ArrayShape<T, DIMS..., N>; using Base::RANK; using Base::VALUE; };

    template<class T>
    struct ArrayShape<T[]> : ArrayShape<T, 0uz> { using Base = ArrayShape<T, 0uz>; using Base::RANK; using Base::VALUE; };

    template <typename T1, typename T2>
    using copyConst = std::conditional_t<std::is_const_v<std::remove_reference_t<T1>>, const T2, T2>;

    template <auto... Xs>
    struct List { static constexpr std::size_t COUNT = sizeof...(Xs); };

    template <char... Cs>
    struct String { static constexpr char STR[] = {Cs..., '\0'}; };

    template <std::size_t SIZE>
    consteval auto sequenceList() {
        return []<std::size_t... IDX>(std::index_sequence<IDX...>&&) consteval {
            return List<IDX...>{};
        }(std::make_index_sequence<SIZE>{});
    }

    template <std::size_t SIZE, auto VALUE, template <auto...> typename CONTAINER = List>
    consteval auto repeatedList() {
        return []<std::size_t... IDX>(std::index_sequence<IDX...>&&) consteval {
            return CONTAINER<((void)IDX, VALUE)...>{};
        }(std::make_index_sequence<SIZE>{});
    }
}
