module;

#include <cstddef> // ptrdiff_t, size_t

export module linalg:pointer;


namespace linalg {
    // Simple pointer type, takes user-supplied pointer and optional stride
    export template <typename T, std::size_t C, std::ptrdiff_t STRIDE = 1z>
    struct PointerClass {
    protected:
        static constexpr std::size_t COUNT = C;
        static constexpr bool ISREF = false;

        constexpr decltype(auto) get(this auto&& self, std::size_t i) { return *(self.data + static_cast<std::ptrdiff_t>(i) * STRIDE); }

    public:
        constexpr PointerClass(T* origin) : data(origin) {}

    protected:
        T* data;
    };
}
