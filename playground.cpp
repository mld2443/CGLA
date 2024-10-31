#include <iostream>


template <std::size_t V> class Carrier {};

template <typename T, std::size_t PROD, std::size_t DIM, std::size_t... REST>
class RType : RType<T, PROD*DIM, REST...> {
protected:
    using BASE = RType<T, PROD*DIM, REST...>;
    using ARRTYPE = typename BASE::ARRTYPE[DIM];

public:
    using BASE::COUNT;

    constexpr RType(ARRTYPE&& payload) : BASE(std::forward<typename BASE::ARRTYPE>(*payload)) {}
};

template <typename T, std::size_t PROD, std::size_t DIM>
class RType<T, PROD, DIM> {
protected:
    using ARRTYPE = T[DIM];

private:
    template <std::size_t... IDX>
    constexpr RType(T&& first, std::index_sequence<IDX...>&&) : data{ (&first)[IDX]... } {}

public:
    static constexpr size_t COUNT = PROD * DIM;

    constexpr RType(ARRTYPE&& payload) : RType<T, PROD, DIM>(std::forward<T>(*payload), std::make_index_sequence<COUNT>{}) {}

protected:
    T data[COUNT];
};
template <typename T, std::size_t D1>
RType(T (&&)[D1]) -> RType<T, 1uz, D1>;
template <typename T, std::size_t D1, std::size_t D2>
RType(T (&&)[D1][D2]) -> RType<T, 1uz, D1, D2>;
template <typename T, std::size_t D1, std::size_t D2, std::size_t D3>
RType(T (&&)[D1][D2][D3]) -> RType<T, 1uz, D1, D2, D3>;
template <typename T, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4>
RType(T (&&)[D1][D2][D3][D4]) -> RType<T, 1uz, D1, D2, D3, D4>;
template <typename T, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5>
RType(T (&&)[D1][D2][D3][D4][D5]) -> RType<T, 1uz, D1, D2, D3, D4, D5>;
template <typename T, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6>
RType(T (&&)[D1][D2][D3][D4][D5][D6]) -> RType<T, 1uz, D1, D2, D3, D4, D5, D6>;
template <typename T, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7>
RType(T (&&)[D1][D2][D3][D4][D5][D6][D7]) -> RType<T, 1uz, D1, D2, D3, D4, D5, D6, D7>;
template <typename T, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7, std::size_t D8>
RType(T (&&)[D1][D2][D3][D4][D5][D6][D7][D8]) -> RType<T, 1uz, D1, D2, D3, D4, D5, D6, D7, D8>;
template <typename T, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7, std::size_t D8, std::size_t D9>
RType(T (&&)[D1][D2][D3][D4][D5][D6][D7][D8][D9]) -> RType<T, 1uz, D1, D2, D3, D4, D5, D6, D7, D8, D9>;


using namespace std;
int main() {
    constexpr auto x = RType({
        {
            { 1.0, 1.0 },
            { 1.0, 1.0 },
            { 1.0, 1.0 }
        }
    });
    cout << x.COUNT << endl;
}
