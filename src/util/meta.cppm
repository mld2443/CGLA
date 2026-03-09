export module meta;

import std;

// Helper Macros for the unused warnings
#if defined(__GNUC__) || defined(__clang__)
#define DISABLE_UNUSED_WARNING _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wunused-value\"")
#define RESTORE_UNUSED_WARNING _Pragma("GCC diagnostic pop")
#elif defined(_MSC_VER)
#define DISABLE_UNUSED_WARNING __pragma(warning(push)) __pragma(warning(disable: 4101))
#define RESTORE_UNUSED_WARNING __pragma(warning(pop))
#else
#define DISABLE_UNUSED_WARNING
#define RESTORE_UNUSED_WARNING
#endif


// Template Metaprogramming
export namespace meta {
    template <typename T1, typename T2>
    using copyConst = std::conditional_t<std::is_const_v<std::remove_reference_t<T1>>, const T2, T2>;

    template <auto...>
    class List {};

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
            DISABLE_UNUSED_WARNING
            return CONTAINER<(IDX, VALUE)...>{};
            RESTORE_UNUSED_WARNING
        }(std::make_index_sequence<SIZE>{});
    }
}
