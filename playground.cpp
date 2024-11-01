#include <iostream>
#include <type_traits>

#define COPYCONSTFORTYPE(T1, ...) std::conditional_t<std::is_const_v<std::remove_reference_t<T1>>, const __VA_ARGS__, __VA_ARGS__>

template <std::ptrdiff_t S, typename T, std::size_t PRODUCT, std::size_t DIM, std::size_t NEXT = 0uz, std::size_t... REST>
class ValueType : public ValueType<S, T, PRODUCT * DIM, NEXT, REST...> {
protected:
    using BASE = ValueType<S, T, PRODUCT * DIM, NEXT, REST...>;
    using ARRTYPE = typename BASE::ARRTYPE[DIM];

public:
    using BASE::COUNT;

    constexpr ValueType(ARRTYPE&& payload) : BASE(std::forward<typename BASE::ARRTYPE>(*payload)) {}
};

template <std::ptrdiff_t S, typename T, std::size_t PRODUCT>
class ValueType<S, T, PRODUCT, 0uz> {
public:
    template <typename POINTERTYPE>
    class Iterator {
    private:
        POINTERTYPE* pos;

    public:
        constexpr Iterator(POINTERTYPE* p) : pos(p) {}

        constexpr decltype(auto)  operator*(this auto& self) { return *self.pos; }
        constexpr decltype(auto) operator++(this auto& self) { self.pos += S; return self; }
        constexpr bool operator==(const Iterator& o) const = default;
    };

protected:
    using ARRTYPE = T;

private:
    template <std::size_t... IDX>
    constexpr ValueType(ARRTYPE&& first, std::index_sequence<IDX...>&&) : data{ (&first)[IDX]... } {}

public:
    static constexpr size_t COUNT = PRODUCT;

    constexpr ValueType(ARRTYPE&& first) : ValueType<S, T, PRODUCT, 0uz>(std::forward<T>(first), std::make_index_sequence<COUNT>{}) {}

    // Iterators for for-each loops
    constexpr auto begin(this auto& self) { return Iterator<COPYCONSTFORTYPE(decltype(self), T)>{ self.data }; }
    constexpr auto   end(this auto& self) { return Iterator<COPYCONSTFORTYPE(decltype(self), T)>{ self.data + static_cast<std::ptrdiff_t>(COUNT) * S }; }

protected:
    T data[COUNT];
};
template <typename T, std::size_t D0>
ValueType(T (&&)[D0]) -> ValueType<1z, T, 1uz, D0>;
template <typename T, std::size_t D0, std::size_t D1>
ValueType(T (&&)[D0][D1]) -> ValueType<1z, T, 1uz, D0, D1>;
template <typename T, std::size_t D0, std::size_t D1, std::size_t D2>
ValueType(T (&&)[D0][D1][D2]) -> ValueType<1z, T, 1uz, D0, D1, D2>;
template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3>
ValueType(T (&&)[D0][D1][D2][D3]) -> ValueType<1z, T, 1uz, D0, D1, D2, D3>;
template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4>
ValueType(T (&&)[D0][D1][D2][D3][D4]) -> ValueType<1z, T, 1uz, D0, D1, D2, D3, D4>;
template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5>
ValueType(T (&&)[D0][D1][D2][D3][D4][D5]) -> ValueType<1z, T, 1uz, D0, D1, D2, D3, D4, D5>;
template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6>
ValueType(T (&&)[D0][D1][D2][D3][D4][D5][D6]) -> ValueType<1z, T, 1uz, D0, D1, D2, D3, D4, D5, D6>;
template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7>
ValueType(T (&&)[D0][D1][D2][D3][D4][D5][D6][D7]) -> ValueType<1z, T, 1uz, D0, D1, D2, D3, D4, D5, D6, D7>;
template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7, std::size_t D8>
ValueType(T (&&)[D0][D1][D2][D3][D4][D5][D6][D7][D8]) -> ValueType<1z, T, 1uz, D0, D1, D2, D3, D4, D5, D6, D7, D8>;
template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7, std::size_t D8, std::size_t D9>
ValueType(T (&&)[D0][D1][D2][D3][D4][D5][D6][D7][D8][D9]) -> ValueType<1z, T, 1uz, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9>;

template <typename T, std::size_t M, std::size_t N>
using Matrix = ValueType<1z, T, 1uz, M, N>;

using namespace std;
int main() {
    constexpr auto x = Matrix({
        { -1.0, 1.0 },
        { 1.0, -1.0 },
        { -1.0, 1.0 }
    });
    cout << x.COUNT << endl;

    for (const auto& e : x)
        cout << e << " ";
    cout << endl;
}
