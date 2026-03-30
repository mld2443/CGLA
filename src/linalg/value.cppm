module;

#include <cstddef>     // size_t
#include <utility>     // forward

export module linalg:value;

import meta;

namespace linalg {
    template <typename T, std::size_t COUNT, std::size_t... REST>
    class RecursiveValueClass;

    // Value type recursive base-case class partial template specialization
    template <typename T, std::size_t COUNT>
    class RecursiveValueClass<T, COUNT> {
    protected:
        using NestedArray = T;

    private:
        template <std::size_t... IDX>
        constexpr RecursiveValueClass(meta::List<IDX...>&&, NestedArray* first) : data{ first[IDX]... } {}

    protected:
        constexpr RecursiveValueClass(NestedArray&& first) : RecursiveValueClass(meta::iotaList<COUNT>(), &first) {}
        constexpr RecursiveValueClass(auto&&... payload) : data{ std::forward<T>(payload)... } {}

        T data[COUNT];
    };

    // Value type recursive primary template
    template <typename T, std::size_t COUNT, std::size_t DIM, std::size_t... REST>
    class RecursiveValueClass<T, COUNT, DIM, REST...> : RecursiveValueClass<T, COUNT, REST...> {
        using Base = RecursiveValueClass<T, COUNT, REST...>;
    protected:
        using NestedArray = typename Base::NestedArray[DIM];
        using Base::data;

    public:
        constexpr RecursiveValueClass(NestedArray&& payload) : Base(std::forward<typename Base::NestedArray>(*payload)) {}
        constexpr RecursiveValueClass(auto&&... payload) : Base(std::forward<T>(payload)...) {}
    };

    // Top-level Value-type class
    export template <typename T, std::size_t... DIMS>
    struct ValueClass : RecursiveValueClass<T, (DIMS * ... * 1uz), DIMS...> {
    protected:
        static constexpr std::size_t COUNT = (DIMS * ... * 1uz);
        static constexpr bool ISREF = false;
    private:
        using Base = RecursiveValueClass<T, COUNT, DIMS...>;

    protected:
        using Base::Base;

        constexpr decltype(auto) get(this auto&& self, std::size_t i) { return *(self.data + i); }
    };
}
