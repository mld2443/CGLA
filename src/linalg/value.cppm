export module linalg:value;

import std;

import meta;

export namespace linalg {
    // Value type recursive primary template
    template <typename T, std::size_t COUNT, std::size_t DIM = 0uz, std::size_t... REST>
    class RecursiveValueClass : RecursiveValueClass<T, COUNT, REST...> {
        using Base = RecursiveValueClass<T, COUNT, REST...>;
    protected:
        using NestedArray = typename Base::NestedArray[DIM];
        using Base::data;

    public:
        constexpr RecursiveValueClass(NestedArray&& payload) : Base(std::forward<typename Base::NestedArray>(*payload)) {}
        constexpr RecursiveValueClass(auto&&... payload) : Base(std::forward<T>(payload)...) {}
    };

    // Value type recursive base-case class partial template specialization
    template <typename T, std::size_t COUNT>
    class RecursiveValueClass<T, COUNT> {
    protected:
        using NestedArray = T;

    private:
        template <std::size_t... IDX>
        constexpr RecursiveValueClass(meta::List<IDX...>&&, NestedArray* first) : data{ first[IDX]... } {}

    protected:
        constexpr RecursiveValueClass(NestedArray&& first) : RecursiveValueClass(meta::sequenceList<COUNT>(), &first) {}
        constexpr RecursiveValueClass(auto&&... payload) : data{ std::forward<T>(payload)... } {}

        T data[COUNT];
    };

    // Top-level Value-type class
    template <typename T, std::size_t... DIMS>
    struct ValueClass : RecursiveValueClass<T, (DIMS * ... * 1uz), DIMS...> {
    protected:
        static constexpr std::size_t COUNT = (DIMS * ... * 1uz);
        static constexpr bool ISREF = false;
    private:
        using Base = RecursiveValueClass<T, COUNT, DIMS...>;

    protected:
        using Base::Base;
        using Base::data;

        constexpr decltype(auto) get(this auto&& self, std::size_t i) { return *(self.data + i); }
    };
}
