module;

#include <cstddef>     // size_t
#include <type_traits> // conditional_t, remove_reference_t
#include <utility>     // forward, index_sequence, make_index_sequence

export module meta;


// Template Metaprogramming
export namespace meta {
    template <class, std::size_t... DIMS>
    struct ArrayTraits {
        static constexpr std::size_t RANK = sizeof...(DIMS);
        static constexpr std::size_t SHAPE[] = { DIMS... };
        static constexpr std::size_t COUNT = (DIMS * ...);
    };

    template<class T, std::size_t N, std::size_t... DIMS>
    struct ArrayTraits<T[N], DIMS...> : ArrayTraits<T, DIMS..., N> {
        using Base = ArrayTraits<T, DIMS..., N>;
        using Base::RANK;
        using Base::SHAPE;
        using Base::COUNT;
    };

    template<class T>
    struct ArrayTraits<T[]> : ArrayTraits<T, 0uz> {
        using Base = ArrayTraits<T, 0uz>;
        using Base::RANK;
        using Base::SHAPE;
        using Base::COUNT;
    };

    template <typename T1, typename T2>
    using copyConst = std::conditional_t<std::is_const_v<std::remove_reference_t<T1>>, const T2, T2>;

    template <auto... Xs>
    struct List {
        static constexpr std::size_t COUNT = sizeof...(Xs);
    };

    template <char... Cs>
    struct String {
        static constexpr std::size_t LENGTH = sizeof...(Cs);
        static constexpr char STR[] = {Cs..., '\0'};
    };

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
