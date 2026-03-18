module;

#include <cstddef>     // size_t
#include <type_traits> // conditional_t, remove_reference_t
#include <utility>     // index_sequence, make_index_sequence

export module meta;


// Template Metaprogramming
export namespace meta {
    template <class, std::size_t... DIMS>
    struct ArrayTraits;

    template <class T, std::size_t... DIMS>
    struct ArrayTraits {
        static constexpr bool        ISARRAY = true;
        static constexpr std::size_t    RANK = sizeof...(DIMS);
        static constexpr std::size_t   COUNT = (DIMS * ...);
        static constexpr std::size_t SHAPE[] = { DIMS... };
    };

    template<class T, std::size_t N, std::size_t... DIMS>
    struct ArrayTraits<T[N], DIMS...> : ArrayTraits<T, DIMS..., N> {
        using Base = ArrayTraits<T, DIMS..., N>;
        using Base::ISARRAY;
        using Base::RANK;
        using Base::COUNT;
        using Base::SHAPE;
    };

    template<class T>
    struct ArrayTraits<T[]> : ArrayTraits<T, 0uz> {
        using Base = ArrayTraits<T, 0uz>;
        using Base::ISARRAY;
        using Base::RANK;
        using Base::COUNT;
        using Base::SHAPE;
    };

    template <class T>
    struct ArrayTraits<T> {
        static constexpr bool        ISARRAY = false;
        static constexpr std::size_t    RANK = 0uz;
        static constexpr std::size_t   COUNT = 1uz;
        static constexpr std::size_t SHAPE[] = { 0uz };
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
