export module util;

import std;

export namespace util {
    template <typename T1, typename T2>
    using copyConst = std::conditional_t<std::is_const_v<std::remove_reference_t<T1>>, const T2, T2>;

    template <typename T1, typename T2>
    constexpr auto cmpLess(T1&& first, T2&& second) {
        if constexpr (requires{ std::is_integral_v<T1> && std::is_integral_v<T2>; })
            return std::cmp_less(std::forward<T1>(first), std::forward<T2>(second)) ? std::forward<T1>(first) : std::forward<T2>(second);
        else
            return first < second ? std::forward<T1>(first) : std::forward<T2>(second);
    }

    template <typename T1, typename T2>
    constexpr auto cmpGreater(T1&& first, T2&& second) {
        if constexpr (requires{ std::is_integral_v<T1> && std::is_integral_v<T2>; })
            return std::cmp_greater(std::forward<T1>(first), std::forward<T2>(second)) ? std::forward<T1>(first) : std::forward<T2>(second);
        else
            return first > second ? std::forward<T1>(first) : std::forward<T2>(second);
    }

    template <typename T1, typename T2, typename... Ts>
    constexpr auto minimum(T1&& first, T2&& second, Ts&&... rest) {
        auto&& smaller = cmpLess(std::forward<T1>(first), std::forward<T2>(second));
        if constexpr (sizeof...(rest)) return minimum(std::forward<decltype(smaller)>(smaller), std::forward<Ts>(rest)...);
        else                           return std::forward<decltype(smaller)>(smaller);
    }

    template <typename T1, typename T2, typename... Ts>
    constexpr auto maximum(T1&& first, T2&& second, Ts&&... rest) {
        auto&& larger = cmpGreater(std::forward<T1>(first), std::forward<T2>(second));
        if constexpr (sizeof...(rest)) return maximum(std::forward<decltype(larger)>(larger), std::forward<Ts>(rest)...);
        else                           return std::forward<decltype(larger)>(larger);
    }

    // ugh, std::transform is such a terrible design pattern, takes flexible containers and transforms *in-place*!
    template <typename FROM, template <typename> typename CONTAINER>
    auto actuallyTransform(CONTAINER<FROM> from, const auto& transform) {
        CONTAINER<decltype(transform(FROM{}))> to;

        if constexpr (requires{ to.reserve(0uz); })
            to.reserve(from.size());

        for (FROM &obj : from)
            to.push_back(transform(obj));

        return to;
    }
}