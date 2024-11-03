#pragma once

#if !__cpp_constexpr
#  error "this compiler does not support constexpr"
#elif !__cpp_if_constexpr
#  error "this compiler does not support constexpr if"
#elif !__cpp_concepts
#  error "this compiler does not support concepts"
#elif !__cpp_decltype_auto
#  error "this compiler does not support decltype auto"
#elif !__cpp_return_type_deduction
#  error "this compiler does not support return type deduction"
#elif !__cpp_inheriting_constructors
#  error "this compiler does not support inherited constructors"
#elif !__cpp_lambdas
#  error "this compiler does not support lambdas"
#elif !__cpp_fold_expressions
#  error "this compiler does not support fold expressions"
#elif !__cpp_nontype_template_args
#  error "this compiler does not support non-type templates"
#elif !__cpp_size_t_suffix
#  error "this compiler does not support size-type suffixes" //for some reason
#elif !__cpp_return_type_deduction
#  error "this compiler does not support return type deduction"
#elif !__cpp_variadic_templates
#  error "this compiler does not support variadic templates"
#elif !__cpp_generic_lambdas
#  error "this compiler does not support generic lambdas"
#elif !__cpp_deduction_guides
#  error "this compiler does not support template deduction guides"
#elif !__cpp_explicit_this_parameter && !(defined(__clang__) && __clang_major__ >= 18) // clang doesn't define this feature test correctly?
#  error "this compiler does not support explicit (deducing) this"
#elif !__cpp_multidimensional_subscript
#  error "this compiler does not support multidimensional subscript"
#endif

#include <cmath>       // sqrt
#include <concepts>    // same_as
#include <cstddef>     // size_t, ptrdiff_t
#include <iostream>    // ostream
#include <type_traits> // conditional_t, is_const_v, remove_reference_t
#include <utility>     // forward, index_sequence, make_index_sequence

namespace linalg {
    // TODO:
    // [x] SETTLE ON PARADIGM: It's multilinear tensors all the way down and recursive inheritance
    // [x] Generic tensor accessor
    // [x] Display arbitrary tensors
    // [x] Shared ops: map(x), fold(x), scalar mult/div(x), negate(x), add(x), subtract(x), maybe inline mult/div(1)
    // [x] Vector ops: dot (x), cross (x)
    // [x] Rewrite Matrix and Vector as type aliases; clang 18 doesn't support template deductions for aliases
    // [x] Merge StorageBase features into Tensor and consider removing it(x)
    // [x] Default initialization(x)
    // [x] Rework "internal" functions into lambdas(x)
    // [x] Merge recursive ValueType into Tensor(x) and ensure base functionality(x)
    // [ ] Matrix transpose(4)
    // [ ] Vector transpose -> Matrix(1)
    // [ ] Matrix ops: matrix mult (2)(add [[nodiscard]] attr), eigenvector/value(3), determinant(3), invert(1), identity(1), rank(3), ...
    // [ ] Tensor ops: tensor product?(?)


    // Helper macros to reduce clutter, undefined at end of namespace
    #define COPYCONSTFORTYPE(T1, ...) std::conditional_t<std::is_const_v<std::remove_reference_t<T1>>, const __VA_ARGS__, __VA_ARGS__>
    #define MAKEINDICES(SIZE) std::make_index_sequence<SIZE>{}

    ///////////////////
    // STORAGE TYPES //
    ///////////////////

    // Shared iterator type for for-each loops
    template <typename QUALIFIEDTYPE, std::ptrdiff_t STRIDE = 1z>
    class Iterator {
    public:
        constexpr Iterator(QUALIFIEDTYPE* p) : pos(p) {}

        constexpr QUALIFIEDTYPE& operator*() { return *pos; }
        constexpr auto operator++() { pos += STRIDE; return *this; }
        constexpr bool operator==(const Iterator& o) const = default;

    private:
        QUALIFIEDTYPE* pos;
    };

    // Value type recursive primary template
    template <typename T, std::size_t PRODUCT, std::size_t DIM = 0uz, std::size_t... REST>
    class ValueTypeRecursive : private ValueTypeRecursive<T, PRODUCT * DIM, REST...> {
        using BASE = ValueTypeRecursive<T, PRODUCT * DIM, REST...>;
    protected:
        using MYTYPE = typename BASE::MYTYPE[DIM];
        using BASE::data;

    public:
        constexpr ValueTypeRecursive(MYTYPE&& payload) : BASE(std::forward<typename BASE::MYTYPE>(*payload)) {}
        constexpr ValueTypeRecursive(auto&&... payload) : BASE(std::forward<T>(payload)...) {}
    };

    // Value type recursive base-case class partial template specialization
    template <typename T, std::size_t COUNT>
    class ValueTypeRecursive<T, COUNT> {
    protected:
        using MYTYPE = T;

    private:
        template <std::size_t... IDX>
        constexpr ValueTypeRecursive(std::index_sequence<IDX...>&&, MYTYPE* first) : data{ first[IDX]... } {} // clang reports past-the-end deref illegal for constexpr

    protected:
        constexpr ValueTypeRecursive(MYTYPE&& first) : ValueTypeRecursive<T, COUNT>(MAKEINDICES(COUNT), &first) {}
        constexpr ValueTypeRecursive(auto&&... payload) : data{ std::forward<T>(payload)... } {}

        T data[COUNT];
    };

    // Top-level Value-type class
    template <typename T, std::size_t... DIMS>
    class ValueType : private ValueTypeRecursive<T, 1uz, DIMS...> {
        using BASE = ValueTypeRecursive<T, 1uz, DIMS...>;
    protected:
        using BASE::ValueTypeRecursive;
        using BASE::data;
        static constexpr std::size_t COUNT = (DIMS * ...);

        // Accessor addressing flat array of data, used internally to perform map
        template <class Self>
        constexpr decltype(auto) get(this Self&& self, std::size_t i) { return *(std::forward<Self>(self).data + i); }

    public:
        // Iterators
        constexpr auto begin(this auto& self) { return Iterator<COPYCONSTFORTYPE(decltype(self), T)>{ self.data }; }
        constexpr auto   end(this auto& self) { return Iterator<COPYCONSTFORTYPE(decltype(self), T)>{ self.data + COUNT }; } // gcc warns if '*end()' would be OOB, even when it isn't dereffed
    };

    // Reference type, transient type with no ref counting
    template <typename T, std::size_t C, std::ptrdiff_t STRIDE = 1z>
    class ReferenceType {
    protected:
        static constexpr std::size_t COUNT = C;

        // Accessor addressing flat array of data, used internally to perform map
        template <class Self>
        constexpr decltype(auto) get(this Self&& self, std::size_t i) { return *(std::forward<Self>(self).data + static_cast<std::ptrdiff_t>(i) * STRIDE); }

    public:
        // Iterators
        constexpr auto begin(this auto& self) { return Iterator<COPYCONSTFORTYPE(decltype(self), T), STRIDE>{ self.data }; }
        constexpr auto   end(this auto& self) { return Iterator<COPYCONSTFORTYPE(decltype(self), T), STRIDE>{ self.data + static_cast<std::ptrdiff_t>(COUNT) * STRIDE }; } // gcc warns if '*end()' would be OOB, even when it isn't dereffed

        constexpr ReferenceType(T* origin) : data(origin) {}

    protected:
        T* data;
    };

    ////////////
    // TENSOR //
    ////////////

    template <class StorageBase, typename T, std::size_t... DIMS>
    class TensorType;

    // Convenience aliases
    template <typename T, std::size_t... DIMS>
    using Tensor = TensorType<ValueType<T, DIMS...>, T, DIMS...>;
    template <typename T, std::size_t M, std::size_t N>
    using Matrix = TensorType<ValueType<T, M, N>, T, M, N>;
    template <typename T, std::size_t M, std::size_t N, std::ptrdiff_t STRIDE = 1z>
    using MatrixRef = TensorType<ReferenceType<T, M * N, STRIDE>, T, M, N>;
    template <typename T, std::size_t N>
    using Vector = TensorType<ValueType<T, N>, T, N>;
    template <typename T, std::size_t N, std::ptrdiff_t STRIDE = 1z>
    using VectorRef = TensorType<ReferenceType<T, N, STRIDE>, T, N>;

    // Deduction guides for value-initialization
    // Anything higher than 10-dimensional can still be value-initialized, but template params must be explicit
    template <typename T, std::size_t D0>
    TensorType(T (&&)[D0]) -> TensorType<ValueType<T, D0>, T, D0>;
    template <typename T, std::size_t D0, std::size_t D1>
    TensorType(T (&&)[D0][D1]) -> TensorType<ValueType<T, D0, D1>, T, D0, D1>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2>
    TensorType(T (&&)[D0][D1][D2]) -> TensorType<ValueType<T, D0, D1, D2>, T, D0, D1, D2>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3>
    TensorType(T (&&)[D0][D1][D2][D3]) -> TensorType<ValueType<T, D0, D1, D2, D3>, T, D0, D1, D2, D3>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4>
    TensorType(T (&&)[D0][D1][D2][D3][D4]) -> TensorType<ValueType<T, D0, D1, D2, D3, D4>, T, D0, D1, D2, D3, D4>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5>
    TensorType(T (&&)[D0][D1][D2][D3][D4][D5]) -> TensorType<ValueType<T, D0, D1, D2, D3, D4, D5>, T, D0, D1, D2, D3, D4, D5>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6>
    TensorType(T (&&)[D0][D1][D2][D3][D4][D5][D6]) -> TensorType<ValueType<T, D0, D1, D2, D3, D4, D5, D6>, T, D0, D1, D2, D3, D4, D5, D6>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7>
    TensorType(T (&&)[D0][D1][D2][D3][D4][D5][D6][D7]) -> TensorType<ValueType<T, D0, D1, D2, D3, D4, D5, D6, D7>, T, D0, D1, D2, D3, D4, D5, D6, D7>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7, std::size_t D8>
    TensorType(T (&&)[D0][D1][D2][D3][D4][D5][D6][D7][D8]) -> TensorType<ValueType<T, D0, D1, D2, D3, D4, D5, D6, D7, D8>, T, D0, D1, D2, D3, D4, D5, D6, D7, D8>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7, std::size_t D8, std::size_t D9>
    TensorType(T (&&)[D0][D1][D2][D3][D4][D5][D6][D7][D8][D9]) -> TensorType<ValueType<T, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9>, T, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9>;

    // TensorType definition
    template <class StorageBase, typename T, std::size_t... DIMS>
    class TensorType : public StorageBase {
    private:
        using StorageBase::COUNT;
        using StorageBase::get;
        template <class OtherBase, typename T2, std::size_t... DIMS2>
        friend class TensorType;
        friend StorageBase;

        // Special 'template container' prettyPrint() uses to build compile-time c-string whitespace
        template <char... STR>
        struct String { static constexpr char VALUES[] = {STR..., '\0'}; };

        // Helper for operator<<, displays arbitrary dimensional tensors in a human-readable format
        template <std::size_t STEP, std::size_t THISDIM, std::size_t NEXTDIM = 0uz, std::size_t... RESTDIMS, char... PRFX, std::size_t... IDX>
        constexpr void prettyPrint(std::ostream& os, std::index_sequence<IDX...>&&, std::size_t offset = 0uz, String<PRFX...> prefix = {}) const {
            auto genSpace = []<std::size_t... IDX2>(std::index_sequence<IDX2...>&&) constexpr { return String<PRFX..., (' ' + static_cast<char>(0uz & IDX2))...>(); };

            constexpr size_t DIMSREMAINING = sizeof...(RESTDIMS) + (NEXTDIM != 0uz) + 1uz;
            if constexpr (DIMSREMAINING > 3uz && DIMSREMAINING % 3uz != 0uz )
                os << (offset ? "\n" : "");

            if constexpr (DIMSREMAINING % 3uz == 0uz)
                (prettyPrint<STEP / NEXTDIM, NEXTDIM, RESTDIMS...>(os, MAKEINDICES(NEXTDIM), offset + IDX * STEP, genSpace(MAKEINDICES(((DIMSREMAINING - 3uz) ? (DIMSREMAINING - 3uz) : 3uz) * IDX))), ...);
            else if constexpr (NEXTDIM)
                (prettyPrint<STEP / NEXTDIM, NEXTDIM, RESTDIMS...>(os, MAKEINDICES(NEXTDIM), offset + IDX * STEP, String<PRFX...>{}), ...);
            else {
                os << (offset ? "\n" : "") << prefix.VALUES;
                ((os << (IDX ? ", " : "") << get(offset + IDX)), ...) << ((offset + THISDIM < COUNT) ? "," : "");
            }
        }

        template <class OtherBase, typename T2, std::size_t... IDX>
        constexpr auto binaryMapInternal(auto func, const TensorType<OtherBase, T2, DIMS...>& v, std::index_sequence<IDX...>) const {
            return TensorType<ValueType<T, DIMS...>, decltype(func(T(), T2())), DIMS...>{ func(get(IDX), v.get(IDX))... };
        }
        template <std::size_t... IDX>
        inline void mapWriteInternal(auto func, std::index_sequence<IDX...>) {
            (func(get(IDX)), ...);
        }
        template <std::size_t... IDX>
        inline void binaryMapWriteInternal(auto func, const auto& v, std::index_sequence<IDX...>) {
            (func(get(IDX), v.get(IDX)), ...);
        }

    public:
        // Constructors
        using StorageBase::StorageBase;

        // Metadata
        constexpr std::size_t count()          const { return COUNT; }
        constexpr std::size_t dimensionality() const { return sizeof...(DIMS); }

        // Functional programming support
        constexpr auto reduce(auto func, auto starting) const {
            return [this]<std::size_t... IDX>(auto& func, auto starting, std::index_sequence<IDX...>) constexpr {
                return ((starting = func(starting, get(IDX))), ...);
            }(func, starting, MAKEINDICES(COUNT));
        }
        constexpr auto reduce(auto func) const {
            return [this]<std::size_t... IDX>(auto& func, T starting, std::index_sequence<IDX...>) constexpr {
                return ((starting = func(starting, get(1uz + IDX))), ...);
            }(func, get(0uz), MAKEINDICES(COUNT - 1uz));
        }
        constexpr auto map(auto func) const {
            return [this]<std::size_t... IDX>(auto func, std::index_sequence<IDX...>) constexpr {
                return TensorType<ValueType<T, DIMS...>, decltype(func(T())), DIMS...>{ func(get(IDX))... };
            }(func, MAKEINDICES(COUNT));
        }

        // Member operator overloads
        constexpr auto operator-()              const { return map([  ](const T& e){ return    -e; }); }
        constexpr auto operator*(const auto& s) const { return map([&s](const T& e){ return e * s; }); }
        constexpr auto operator/(const auto& s) const { return map([&s](const T& e){ return e / s; }); }

        // Accessor
        template <class Self>
        constexpr decltype(auto) operator[](this Self&& self, auto first, auto... inds) requires (sizeof...(inds) < sizeof...(DIMS)) {
            return []<std::size_t STEP, std::size_t NEXTDIM, std::size_t... RESTDIMS>(this auto getTensor, Self&& self, std::ptrdiff_t offset, auto nextInd, auto... restInds) constexpr {
                constexpr std::size_t THISSTEP = STEP / NEXTDIM;
                if constexpr (std::is_same_v<decltype(nextInd), char>) { // nextInd is a wildcard
                    if constexpr (sizeof...(restInds))      // there are more indices after this wildcard
                        return 666;
                    else                                    // trailing wildcards do nothing
                        return std::forward<Self>(self);
                } else {
                    offset += THISSTEP * static_cast<std::size_t>(nextInd);
                    if constexpr (sizeof...(restInds))      // more constraints to get through
                        return getTensor.template operator()<THISSTEP, RESTDIMS...>(std::forward<Self>(self), offset, restInds...);
                    else if constexpr (sizeof...(RESTDIMS)) // there are unconstrained dimensions TODO reuse something rather than fold?
                        return TensorType<ReferenceType<COPYCONSTFORTYPE(Self, T), (RESTDIMS * ...)>, COPYCONSTFORTYPE(Self, T), RESTDIMS...>{ std::forward<Self>(self).data + offset };
                    else                                    // the final dimension is constrained, return the value
                        return *(std::forward<Self>(self).data + offset); //TODO need to use storagetype here
                }
            }.template operator()<COUNT, DIMS...>(std::forward<Self>(self), 0z, first, inds...);
        }

        // Member operator overloads
        template <class OtherBase, typename T2>
        constexpr auto operator+(const TensorType<OtherBase, T2, DIMS...>& t) const { return binaryMapInternal([](const T& e1, const T2& e2){ return e1 + e2; }, t, MAKEINDICES(COUNT)); }
        template <class OtherBase, typename T2>
        constexpr auto operator-(const TensorType<OtherBase, T2, DIMS...>& t) const { return binaryMapInternal([](const T& e1, const T2& e2){ return e1 - e2; }, t, MAKEINDICES(COUNT)); }

        // Mutating operators
        inline auto& operator*=(const auto& s) { this->mapWriteInternal([&s](T& e){ e *= s; }, MAKEINDICES(COUNT)); return *this; }
        inline auto& operator/=(const auto& s) { this->mapWriteInternal([&s](T& e){ e /= s; }, MAKEINDICES(COUNT)); return *this; }
        template <class OtherBase, typename T2>
        inline auto& operator+=(const TensorType<OtherBase, T2, DIMS...>& t) { this->binaryMapWriteInternal([](T& e1, const T2& e2){ e1 += e2; }, t, MAKEINDICES(COUNT)); return *this; }
        template <class OtherBase, typename T2>
        inline auto& operator-=(const TensorType<OtherBase, T2, DIMS...>& t) { this->binaryMapWriteInternal([](T& e1, const T2& e2){ e1 -= e2; }, t, MAKEINDICES(COUNT)); return *this; }
        template <class OtherBase, typename T2>
        inline auto&  operator=(const TensorType<OtherBase, T2, DIMS...>& t) { this->binaryMapWriteInternal([](T& e1, const T2& e2){ e1  = e2; }, t, MAKEINDICES(COUNT)); return *this; }

        template <class STORAGETYPE2, typename T2, std::size_t FIRSTDIM, std::size_t... RESTDIMS>
        friend constexpr std::ostream& operator<<(std::ostream& os, const TensorType<STORAGETYPE2, T2, FIRSTDIM, RESTDIMS...>& t);

        /////////////////////
        // SPECIALIZATIONS //
        /////////////////////

        // Vector Geometric methods

        template <class OtherBase, typename T2> requires(sizeof...(DIMS) == 1uz)
        constexpr T dot(this const TensorType<StorageBase, T, COUNT>& self, const TensorType<OtherBase, T2, COUNT>& v) {
            return []<std::size_t... IDX>(const auto& v1, const auto& v2, std::index_sequence<IDX...>) constexpr {
                return ((v1[IDX] * v2[IDX]) + ...);
            }(self, v, MAKEINDICES(COUNT));
        }
        constexpr T magnitudeSqr(this const TensorType<StorageBase, T, COUNT>& self) { return self.dot(self);                 }
        constexpr T    magnitude(this const TensorType<StorageBase, T, COUNT>& self) { return std::sqrt(self.magnitudeSqr()); }
        constexpr auto direction(this const TensorType<StorageBase, T, COUNT>& self) { return self / self.magnitude();        }

        template <class OtherBase, typename T2> requires(sizeof...(DIMS) == 1uz, COUNT == 3uz)
        constexpr TensorType<ValueType<T, 3uz>, decltype(T()*T2()), 3uz> cross(this const TensorType<StorageBase, T, 3uz>& self, const TensorType<OtherBase, T2, 3uz>& v) {
            return { self[1uz]*v[2uz] - self[2uz]*v[1uz],
                     self[2uz]*v[0uz] - self[0uz]*v[2uz],
                     self[0uz]*v[1uz] - self[1uz]*v[0uz] };
        }
    };

    // Right-side operator overloads
    template <class StorageBase, typename T, std::size_t... DIMS>
    constexpr auto operator*(const T& s, const TensorType<StorageBase, T, DIMS...> &t) { return t.map([&s](const T& e) { return e * s; }); }
    template <class StorageBase, typename T, std::size_t... DIMS>
    constexpr auto operator/(const T& s, const TensorType<StorageBase, T, DIMS...> &t) { return t.map([&s](const T& e) { return e / s; }); }
    template <class StorageBase, typename T, std::size_t FIRSTDIM, std::size_t... RESTDIMS>
    constexpr std::ostream& operator<<(std::ostream& os, const TensorType<StorageBase, T, FIRSTDIM, RESTDIMS...>& t) {
        t.template prettyPrint<(RESTDIMS * ... * 1uz), FIRSTDIM, RESTDIMS...>(os, MAKEINDICES(FIRSTDIM));
        return os;
    }

    #undef COPYCONSTFORTYPE
    #undef MAKEINDICES
}

#if 0
template <typename T, std::size_t M, std::size_t N, std::ptrdiff_t S, STORAGECLASS STORAGETYPE>
class MatrixBase : public STORAGETYPE<T, M, N, S>, public TensorRoot<T, M, N> {
private:
    using BaseType = T;
    template <typename TYPE>
    using ReturnType = MatrixBase<TYPE, M, N, 1z, ValueType>;

    template <std::size_t... IDX>
    constexpr static ReturnType<T> matrixIdentity(std::index_sequence<IDX...>) {
        return { (IDX % (M + 1uz) ? T(0) : T(1))... };
    }
    template <typename T2, std::size_t O, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE, std::size_t... IDX>
    constexpr auto matrixMultiply(const MatrixBase<T2, N, O, S2, OTHERSTORAGE>& m, std::index_sequence<IDX...>) const {
        return MatrixBase<decltype(T()*T2()), M, O, 1z, ValueType>{ getRow(IDX / O).dot(m.getCol(IDX % O))... };
    }

public:
    // Identity matrix for some reason
    constexpr static auto I() requires(M == N) { return matrixIdentity(MAKEINDICES(M*N)); }

    // Accessors
    constexpr decltype(auto) operator[](this auto& self, std::size_t m, std::size_t n) { return self.get(n + m*N); }
    template <class MYTYPE> constexpr auto getRow(this MYTYPE& self, std::size_t row) { return VectorBase<COPYCONSTFORTYPE(MYTYPE, T), N,     S, ReferenceType>{ self.data, static_cast<std::ptrdiff_t>(row * N * S) }; }
    template <class MYTYPE> constexpr auto getCol(this MYTYPE& self, std::size_t col) { return VectorBase<COPYCONSTFORTYPE(MYTYPE, T), M, N * S, ReferenceType>{ self.data, static_cast<std::ptrdiff_t>(col     * S) }; }
    template <class MYTYPE> constexpr auto getDiagonal(this MYTYPE& self) { return VectorBase<COPYCONSTFORTYPE(MYTYPE, T), std::min(M, N), (N + 1z) * S, ReferenceType>{ self.data, 0z }; }

    // Member operators
    template<typename T2, std::size_t O, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE>
    constexpr auto operator*(const MatrixBase<T2, N, O, S2, OTHERSTORAGE>& m) const { return matrixMultiply(m, MAKEINDICES(M*O)); }
};
#endif
