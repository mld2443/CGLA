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
#include <concepts>    // same_as, convertible_to
#include <cstddef>     // size_t, ptrdiff_t
#include <iostream>    // ostream
#include <type_traits> // conditional_t, is_const_v, remove_reference_t
#include <utility>     // forward, index_sequence, make_index_sequence


namespace linalg {
    // TODO:
    // [x] SETTLE ON PARADIGM: It's multilinear tensors all the way down
    // [x] Generic tensor accessor
    // [x] Display arbitrary tensors
    // [x] Shared ops: map(x), fold(x), scalar mult/div(x), negate(x), add(x), subtract(x), maybe inline mult/div(1)
    // [x] Vector ops: dot (x), cross (x)
    // [ ] Merge StorageBase into Tensor and remove some of the CRTP-ness
    // [ ] Matrix transpose(3)
    // [ ] Vector transpose -> Matrix(2)
    // [ ] Matrix ops: invert(1), determinant(3), identity(1), rank(3), matrix mult (2)(add [[nodiscard]] attr) ...
    // [ ] Tensor ops: tensor product?!? (6?)


    // Helper macros to reduce clutter, undefined at end of namespace
    #define COPYCONSTFORTYPE(T1, ...) std::conditional_t<std::is_const_v<std::remove_reference_t<T1>>, const __VA_ARGS__, __VA_ARGS__>
    #define STORAGECLASS template <std::ptrdiff_t, typename, std::size_t...> class
    #define MAKEINDICES(SIZE) std::make_index_sequence<SIZE>{}

    ///////////////////
    // STORAGE TYPES //
    ///////////////////

    template <std::ptrdiff_t S, typename T, std::size_t... DIMS>
    class StorageBase {
    public:
        class Iterator {
        private:
            mutable T* pos;

        public:
            constexpr Iterator(T* p) : pos(p) {}

            constexpr decltype(auto)  operator*(this auto& self) { return *self.pos; }
            constexpr decltype(auto) operator++(this auto& self) { self.pos += S; return self; }
            constexpr bool operator==(const Iterator& o) const = default;
        };

        static constexpr std::size_t COUNT = (DIMS * ...);

        static constexpr std::size_t count() { return COUNT; }

        // Iterators for for-each loops
        constexpr auto begin(this auto& self) -> COPYCONSTFORTYPE(decltype(self), Iterator) { return { self.data }; }
        constexpr auto   end(this auto& self) -> COPYCONSTFORTYPE(decltype(self), Iterator) { return { self.data + static_cast<std::ptrdiff_t>(COUNT) * S }; }

    protected:
        // Accessor
        template <class SELF>
        constexpr decltype(auto) get(this SELF&& self, std::size_t i) { return *(std::forward<SELF>(self).data + static_cast<std::ptrdiff_t>(i) * S); }
    };

    // Value type that owns its own data
    template <std::ptrdiff_t, typename T, std::size_t... DIMS>
    class ValueType : public StorageBase<1z, T, DIMS...> {
        friend StorageBase<1z, T, DIMS...>;
    public:
        using StorageBase<1z, T, DIMS...>::COUNT;

        template <std::same_as<T>... Ts> requires((sizeof...(Ts) == 0uz || sizeof...(Ts) == COUNT))
        constexpr ValueType(const Ts&&... payload) : data{ std::forward<const T>(payload)... } {}

    protected:
        T data[COUNT];
    };

    // Reference-type that points to data (no ref counting!)
    //   These should be treated as transient, kinda like an r-value
    template <std::ptrdiff_t S, typename T, std::size_t... DIMS>
    class ReferenceType : public StorageBase<S, T, DIMS...> {
        friend StorageBase<S, T, DIMS...>;

    public:
        constexpr ReferenceType(T* origin, std::ptrdiff_t offset = 0z) : data(origin + offset) {}

    protected:
        T* data;
    };


    //////////////////
    // TENSOR TYPES //
    //////////////////

    // Multilinear tensor
    template <STORAGECLASS STORAGETYPE, std::ptrdiff_t S, typename T, std::size_t... DIMS>
    class Tensor : public STORAGETYPE<S, T, DIMS...> {
        using StorageBase<S, T, DIMS...>::COUNT;
        template <STORAGECLASS STORAGETYPE2, std::ptrdiff_t S2, typename T2, std::size_t... DIMS2>
        friend class Tensor;
        template <typename T2, std::size_t N, STORAGECLASS STORAGETYPE2, std::ptrdiff_t S2>
        friend class Vector;

    private:
        // Special 'template container' prettyPrint() uses to build compile-time c-strings
        template <char... STR>
        struct String {
            static constexpr char VALUES[] = {STR..., '\0'};
        };

        // Displays arbitrary dimensional tensors in a human-readable format
        template <std::size_t STEP, std::size_t THISDIM, std::size_t NEXTDIM = 0uz, std::size_t... RESTDIMS, char... PRFX, std::size_t... IDX>
        constexpr void prettyPrint(std::ostream& os, std::index_sequence<IDX...>&&, std::size_t offset = 0uz, String<PRFX...> prefix = {}) const {
            auto getString = []<std::size_t... IDX2>(std::index_sequence<IDX2...>&&) constexpr { return String<PRFX..., (' ' + static_cast<char>(0uz & IDX2))...>(); };

            constexpr size_t DIMSREMAINING = sizeof...(RESTDIMS) + (NEXTDIM != 0uz) + 1uz;
            if constexpr (DIMSREMAINING > 3uz && DIMSREMAINING % 3uz != 0uz )
                os << (offset ? "\n" : "");

            if constexpr (DIMSREMAINING % 3uz == 0uz)
                (prettyPrint<STEP / NEXTDIM, NEXTDIM, RESTDIMS...>(os, MAKEINDICES(NEXTDIM), offset + IDX * STEP, getString(MAKEINDICES(((DIMSREMAINING - 3uz) ? (DIMSREMAINING - 3uz) : 3uz) * IDX))), ...);
            else if constexpr (NEXTDIM)
                (prettyPrint<STEP / NEXTDIM, NEXTDIM, RESTDIMS...>(os, MAKEINDICES(NEXTDIM), offset + IDX * STEP, String<PRFX...>{}), ...);
            else {
                os << (offset ? "\n" : "") << prefix.VALUES;
                ((os << (IDX ? ", " : "") << this->get(offset + IDX)), ...) << ((offset + THISDIM < COUNT) ? "," : "");
            }
        }

        template <class SELF, std::size_t STEP, std::size_t NEXTDIM, std::size_t... RESTDIMS>
        constexpr decltype(auto) getTensor(this SELF&& self, std::ptrdiff_t offset, auto nextInd, auto... restInds) {
            constexpr std::size_t THISSTEP = STEP / NEXTDIM;
            offset += THISSTEP * static_cast<std::size_t>(nextInd);
            if constexpr (sizeof...(restInds))
                return std::forward<SELF>(self).template getTensor<SELF, THISSTEP, RESTDIMS...>(offset, restInds...);
            else if constexpr (sizeof...(RESTDIMS))
                return Tensor<ReferenceType, S, COPYCONSTFORTYPE(SELF, T), RESTDIMS...>{std::forward<SELF>(self).data, offset};
            else
                return *(std::forward<SELF>(self).data + offset * S);
        }

    protected:
        template <std::size_t... IDX>
        constexpr auto foldInternal(this const auto& self, auto& func, T starting, std::index_sequence<IDX...>) {
            return ((starting = func(starting, self.get(IDX))), ...);
        }
        template <typename MYTYPE, std::size_t... IDX>
        constexpr auto mapInternal(this const MYTYPE& self, auto func, std::index_sequence<IDX...>) {
            return typename MYTYPE::template ReturnType<decltype(func(T()))>{ func(self.get(IDX))... };
        }
        template <typename MYTYPE, STORAGECLASS STORAGETYPE2, std::ptrdiff_t S2, typename T2, std::size_t... IDX>
        constexpr auto binaryMapInternal(this const MYTYPE& self, auto func, const Tensor<STORAGETYPE2, S2, T2, DIMS...>& v, std::index_sequence<IDX...>) {
            return typename MYTYPE::template ReturnType<decltype(func(T(), T2()))>{ func(self.get(IDX), v.get(IDX))... };
        }
        template <std::size_t... IDX>
        inline void mapWriteInternal(this auto& self, auto func, std::index_sequence<IDX...>) {
            (func(self.get(IDX)), ...);
        }
        template <std::size_t... IDX>
        inline void binaryMapWriteInternal(this auto& self, auto func, const auto& v, std::index_sequence<IDX...>) {
            (func(self.get(IDX), v.get(IDX)), ...);
        }

    public:
        // Constructors
        using STORAGETYPE<S, T, DIMS...>::STORAGETYPE;

        // Functional programming support
        constexpr auto fold(this const auto& self, auto func, T starting) { return self.foldInternal(func, starting, MAKEINDICES(COUNT)); }
        constexpr auto map(this const auto& self, auto func) { return self.mapInternal(func, MAKEINDICES(COUNT)); }

        // Member operator overloads
        constexpr auto operator-(this const auto& self)                { return self.map([  ](const T& e){ return    -e; }); }
        constexpr auto operator*(this const auto& self, const auto& s) { return self.map([&s](const T& e){ return e * s; }); }
        constexpr auto operator/(this const auto& self, const auto& s) { return self.map([&s](const T& e){ return e / s; }); }

        // Accessor
        template <class SELF>
        constexpr decltype(auto) operator[](this SELF&& self, auto first, auto... inds) requires (sizeof...(inds) < sizeof...(DIMS)) {
            return std::forward<SELF>(self).template getTensor<SELF, COUNT, DIMS...>(0z, first, inds...);
        }

        // Member operator overloads
        template <STORAGECLASS OTHERSTORAGE, std::ptrdiff_t S2, typename T2>
        constexpr auto operator+(this const auto& self, const Tensor<OTHERSTORAGE, S2, T2, DIMS...>& t) { return self.binaryMapInternal([](const T& e1, const T2& e2){ return e1 + e2; }, t, MAKEINDICES(COUNT)); }
        template <STORAGECLASS OTHERSTORAGE, std::ptrdiff_t S2, typename T2>
        constexpr auto operator-(this const auto& self, const Tensor<OTHERSTORAGE, S2, T2, DIMS...>& t) { return self.binaryMapInternal([](const T& e1, const T2& e2){ return e1 - e2; }, t, MAKEINDICES(COUNT)); }

        // Mutating operators
        inline auto& operator*=(const auto& s) { this->mapWriteInternal([&s](T& e){ e *= s; }, MAKEINDICES(COUNT)); return *this; }
        inline auto& operator/=(const auto& s) { this->mapWriteInternal([&s](T& e){ e /= s; }, MAKEINDICES(COUNT)); return *this; }
        template <STORAGECLASS OTHERSTORAGE, std::ptrdiff_t S2, typename T2>
        inline auto& operator+=(const Tensor<OTHERSTORAGE, S2, T2, DIMS...>& t) { this->binaryMapWriteInternal([](T& e1, const T2& e2){ e1 += e2; }, t, MAKEINDICES(COUNT)); return *this; }
        template <STORAGECLASS OTHERSTORAGE, std::ptrdiff_t S2, typename T2>
        inline auto& operator-=(const Tensor<OTHERSTORAGE, S2, T2, DIMS...>& t) { this->binaryMapWriteInternal([](T& e1, const T2& e2){ e1 -= e2; }, t, MAKEINDICES(COUNT)); return *this; }
        template <STORAGECLASS OTHERSTORAGE, std::ptrdiff_t S2, typename T2>
        inline auto&  operator=(const Tensor<OTHERSTORAGE, S2, T2, DIMS...>& t) { this->binaryMapWriteInternal([](T& e1, const T2& e2){ e1  = e2; }, t, MAKEINDICES(COUNT)); return *this; }

        template <STORAGECLASS STORAGETYPE2, std::ptrdiff_t S2, typename T2, std::size_t FIRSTDIM, std::size_t... RESTDIMS>
        friend constexpr std::ostream& operator<<(std::ostream& os, const Tensor<STORAGETYPE2, S2, T2, FIRSTDIM, RESTDIMS...>& t);
    };

    // Right-side operator overload friends
    template <STORAGECLASS STORAGETYPE, std::ptrdiff_t S, typename T, std::size_t... DIMS>
    constexpr auto operator*(const T& s, const Tensor<STORAGETYPE, S, T, DIMS...> &t) { return t.map([&s](const T& e) { return e * s; }); }
    template <STORAGECLASS STORAGETYPE, std::ptrdiff_t S, typename T, std::size_t... DIMS>
    constexpr auto operator/(const T& s, const Tensor<STORAGETYPE, S, T, DIMS...> &t) { return t.map([&s](const T& e) { return e / s; }); }
    template <STORAGECLASS STORAGETYPE, std::ptrdiff_t S, typename T, std::size_t FIRSTDIM, std::size_t... RESTDIMS>
    constexpr std::ostream& operator<<(std::ostream& os, const Tensor<STORAGETYPE, S, T, FIRSTDIM, RESTDIMS...>& t) {
        t.template prettyPrint<(RESTDIMS * ... * 1uz), FIRSTDIM, RESTDIMS...>(os, MAKEINDICES(FIRSTDIM));
        return os;
    }

    // 1-dimensional vector
    template <typename T, std::size_t N, STORAGECLASS STORAGETYPE = ValueType, std::ptrdiff_t S = 1z>
    class Vector : public Tensor<STORAGETYPE, S, T, N> {
    private:
        template <STORAGECLASS STORAGETYPE2, std::ptrdiff_t S2, typename T2, std::size_t... DIMS2>
        friend class Tensor;
        template <typename TYPE, STORAGECLASS STORAGETYPE2 = ValueType, std::ptrdiff_t S2 = 1z>
        using ReturnType = Vector<TYPE, N, STORAGETYPE2, S2>;

        template <typename T2, STORAGECLASS OTHERSTORAGE, std::ptrdiff_t S2, std::size_t... IDX>
        constexpr auto dotInternal(this const auto& self, const Vector<T2, N, OTHERSTORAGE, S2>& v, std::index_sequence<IDX...>) {
            return ((self[IDX] * v[IDX]) + ...);
        }

    public:
        // Constructors
        using Tensor<STORAGETYPE, S, T, N>::Tensor;
        template <STORAGECLASS OTHERSTORAGE>
        constexpr Vector(Tensor<OTHERSTORAGE, S, T, N>& t) : Tensor<ReferenceType, S, T, N>(t.data) {}
        template <STORAGECLASS OTHERSTORAGE>
        constexpr Vector(const Tensor<OTHERSTORAGE, S, std::remove_const_t<T>, N>& t) : Tensor<ReferenceType, S, T, N>(t.data) {}

        // Geometric methods
        template <typename T2, STORAGECLASS OTHERSTORAGE, std::ptrdiff_t S2>
        constexpr T dot(const Vector<T2, N, OTHERSTORAGE, S2>& v) const { return dotInternal(v, MAKEINDICES(N)); }
        constexpr T magnitudeSqr() const { return dot(*this);                }
        constexpr T    magnitude() const { return std::sqrt(magnitudeSqr()); }
        constexpr auto direction() const { return *this / magnitude();       }

        // Cross product for 3-dimensional vectors
        template <typename T2, STORAGECLASS OTHERSTORAGE, std::ptrdiff_t S2>
        constexpr ReturnType<decltype(T()*T2())> cross(const Vector<T2, 3uz, OTHERSTORAGE, S2>& v) const {
            return { (*this)[1uz]*v[2uz] - (*this)[2uz]*v[1uz],
                     (*this)[2uz]*v[0uz] - (*this)[0uz]*v[2uz],
                     (*this)[0uz]*v[1uz] - (*this)[1uz]*v[0uz] };
        }
    };
    // Template deduction guide to automatically deduce N from initializer lists of rvalues, excluding lvalues
    template <typename T, std::same_as<T>... Ts>
    Vector(const T&&, const Ts&&...) -> Vector<T, 1uz + sizeof...(Ts)>;
    template <STORAGECLASS STORAGETYPE, std::ptrdiff_t S, typename T, std::size_t N>
    Vector(Tensor<STORAGETYPE, S, T, N>& t) -> Vector<T, N, ReferenceType, S>;
    template <STORAGECLASS STORAGETYPE, std::ptrdiff_t S, typename T, std::size_t N>
    Vector(const Tensor<STORAGETYPE, S, T, N>& t) -> Vector<const T, N, ReferenceType, S>;

    // Vector reference-type struct
    template <typename T, std::size_t N, std::ptrdiff_t S = 1z>
    using VectorRef = Vector<T, N, ReferenceType, S>;

    // 2-dimensional matrix
    template <typename T, std::size_t M, std::size_t N, STORAGECLASS STORAGETYPE = ValueType, std::ptrdiff_t S = 1z>
    class Matrix : public Tensor<STORAGETYPE, S, T, M, N> {
    private:
        template <std::size_t... IDX>
        constexpr Matrix(T (&&payload)[M][N], std::index_sequence<IDX...>) : Tensor<ValueType, S, T, M, N>(std::forward<T>(payload[IDX/N][IDX%N])...) {}

    public:
        // Value-initialization constructor
        constexpr Matrix(T (&&payload)[M][N]) : Matrix(std::forward<T[M][N]>(payload), MAKEINDICES(M*N)) {}
    };

    #undef COPYCONSTFORTYPE
    #undef STORAGECLASS
    #undef MAKEINDICES
}

#if 0
namespace legacy {
    // Helper macros to reduce clutter, undefined at end of namespace
    #define COPYCONSTFORTYPE(T1, T2) std::conditional_t<std::is_const_v<T1>, const T2, T2>
    #define STORAGECLASS template <typename, std::size_t, std::size_t, std::ptrdiff_t> class
    #define MAKEINDICES(SIZE) std::make_index_sequence<SIZE>{}

    ////////////////
    // ROOT TYPES //
    ////////////////

    template <typename T, std::size_t M, std::size_t N, std::ptrdiff_t S>
    class StorageRoot {
    public:
        class Iterator {
        private:
            mutable T* pos;

        public:
            constexpr Iterator(T* p) : pos(p) {}

            constexpr decltype(auto)  operator*(this auto& self) { return *self.pos; }
            constexpr decltype(auto) operator++(this auto& self) { self.pos += S; return self; }
            constexpr bool operator==(const Iterator& o) const = default;
        };

    public:
        // Iterators for for-each loops
        constexpr auto begin(this auto& self) -> COPYCONSTFORTYPE(decltype(self), Iterator) { return { self.data }; }
        constexpr auto   end(this auto& self) -> COPYCONSTFORTYPE(decltype(self), Iterator) { return { self.data + static_cast<std::ptrdiff_t>(M*N) * S }; }

    protected:
        // Accessor
        constexpr decltype(auto) get(this auto& self, std::size_t i) { return self.data[static_cast<std::ptrdiff_t>(i) * S]; }
    };

    template <typename T, std::size_t M, std::size_t N>
    class TensorRoot {
    protected:
        template <std::size_t... IDX>
        constexpr auto foldInternal(this const auto& self, auto& func, T starting, std::index_sequence<IDX...>) {
            return ((starting = func(starting, self.get(IDX))), ...);
        }
        template <typename MYTYPE, std::size_t... IDX>
        constexpr auto mapInternal(this const MYTYPE& self, auto func, std::index_sequence<IDX...>) {
            return typename MYTYPE::template ReturnType<decltype(func(T()))>{ func(self.get(IDX))... };
        }
        template <typename MYTYPE, typename OTHER, std::size_t... IDX>
        constexpr auto binaryMapInternal(this const MYTYPE& self, auto func, const OTHER& v, std::index_sequence<IDX...>) {
            return typename MYTYPE::template ReturnType<decltype(func(T(), typename OTHER::BaseType()))>{ func(self.get(IDX), v.get(IDX))... };
        }
        template <std::size_t... IDX>
        inline void mapWriteInternal(this auto& self, auto func, std::index_sequence<IDX...>) {
            (func(self.get(IDX)), ...);
        }
        template <std::size_t... IDX>
        inline void binaryMapWriteInternal(this auto& self, auto func, const auto& v, std::index_sequence<IDX...>) {
            (func(self.get(IDX), v.get(IDX)), ...);
        }

    public:
        constexpr auto fold(this const auto& self, auto func, T starting) { return self.foldInternal(func, starting, MAKEINDICES(M*N)); }
        constexpr auto map(this const auto& self, auto func) { return self.mapInternal(func, MAKEINDICES(M*N)); }

        // Member operator overloads
        constexpr auto operator-(this const auto& self)                { return self.map([  ](auto&    e){ return    -e; }); }
        constexpr auto operator*(this const auto& self, const auto& s) { return self.map([&s](const T& e){ return e * s; }); }
        constexpr auto operator/(this const auto& self, const auto& s) { return self.map([&s](const T& e){ return e / s; }); }
    };


    ///////////////////
    // STORAGE TYPES //
    ///////////////////

    // Value type that owns its own data
    template <typename T, std::size_t M, std::size_t N, std::ptrdiff_t>
    class ValueType : public StorageRoot<T, M, N, 1z> {
        friend StorageRoot<T, M, N, 1z>;
    public:
        template <std::same_as<T>... Ts> requires(sizeof...(Ts) == 0uz || sizeof...(Ts) == M*N)
        constexpr ValueType(Ts&&... payload) : data{ payload... } {}

    protected:
        T data[M*N];
    };

    // Reference-type that points to data (no ref counting!)
    //   These should be treated as transient, kinda like an r-value
    template <typename T, std::size_t M, std::size_t N, std::ptrdiff_t S>
    class ReferenceType : public StorageRoot<T, M, N, S> {
        friend StorageRoot<T, M, N, S>;
    public:
        constexpr ReferenceType(T* origin, std::ptrdiff_t offset) : data(origin + offset) {}

    protected:
        T* data;
    };


    //////////////////
    // TENSOR TYPES //
    //////////////////

    // VECTOR
    template <typename T, std::size_t N, std::ptrdiff_t S, STORAGECLASS STORAGETYPE>
    class VectorBase : public STORAGETYPE<T, N, 1uz, S>, public TensorRoot<T, N, 1uz> {
    private:
        template <typename, std::size_t, std::size_t>
        friend class TensorRoot;

        using BaseType = T;
        template <typename TYPE>
        using ReturnType = VectorBase<TYPE, N, 1z, ValueType>;

        // Implementation for the "broadcast" constructor, curious hack to coax the expansion but discard the values
        template <std::size_t... IDX>
        constexpr VectorBase(const T& value, std::index_sequence<IDX...>) : STORAGETYPE<T, N, 1uz, 1z>(value + T(0uz & IDX)...) {}

        template <typename T2, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE, std::size_t... IDX>
        constexpr auto dotInternal(this const auto& self, const VectorBase<T2, N, S2, OTHERSTORAGE>& v, std::index_sequence<IDX...>) {
            return ((self[IDX] * v[IDX]) + ...);
        }

    public:
        // Constructors
        using STORAGETYPE<T, N, 1uz, S>::STORAGETYPE;
        VectorBase(const T& value) : VectorBase(value, MAKEINDICES(N)) {}

        // Accessor
        constexpr decltype(auto) operator[](this auto& self, std::size_t i) { return self.get(i); }

        // Member operator overloads
        template <typename T2, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE>
        constexpr auto operator+(const VectorBase<T2, N, S2, OTHERSTORAGE>& v) const { return this->binaryMapInternal([](const T& e1, const T2& e2){ return e1 + e2; }, v, MAKEINDICES(N)); }
        template <typename T2, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE>
        constexpr auto operator-(const VectorBase<T2, N, S2, OTHERSTORAGE>& v) const { return this->binaryMapInternal([](const T& e1, const T2& e2){ return e1 - e2; }, v, MAKEINDICES(N)); }

        // Mutating operators
        inline auto& operator*=(const auto& s) { this->mapWriteInternal([&s](T& e){ e *= s; }, MAKEINDICES(N)); return *this; }
        inline auto& operator/=(const auto& s) { this->mapWriteInternal([&s](T& e){ e /= s; }, MAKEINDICES(N)); return *this; }
        template <typename T2, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE>
        inline auto& operator+=(const VectorBase<T2, N, S2, OTHERSTORAGE>& v) { this->binaryMapWriteInternal([](T& e1, const T2& e2){ e1 += e2; }, v, MAKEINDICES(N)); return *this; }
        template <typename T2, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE>
        inline auto& operator-=(const VectorBase<T2, N, S2, OTHERSTORAGE>& v) { this->binaryMapWriteInternal([](T& e1, const T2& e2){ e1 -= e2; }, v, MAKEINDICES(N)); return *this; }
        template <typename T2, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE>
        inline auto&  operator=(const VectorBase<T2, N, S2, OTHERSTORAGE>& v) { this->binaryMapWriteInternal([](T& e1, const T2& e2){ e1  = e2; }, v, MAKEINDICES(N)); return *this; }

        // Geometric methods
        template <typename T2, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE>
        constexpr T dot(const VectorBase<T2, N, S2, OTHERSTORAGE>& v) const { return dotInternal(v, MAKEINDICES(N)); }
        constexpr T magnitudeSqr() const { return dot(*this);                }
        constexpr T    magnitude() const { return std::sqrt(magnitudeSqr()); }
        constexpr auto direction() const { return *this / magnitude();       }

        // Cross product for 3-dimensional vectors
        template <typename T2, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE>
        constexpr ReturnType<decltype(T()*T2())> cross(this const VectorBase<T, 3uz, S, STORAGETYPE>& self, const VectorBase<T2, 3uz, S2, OTHERSTORAGE>& v) {
            return { self[1uz]*v[2uz] - self[2uz]*v[1uz],
                     self[2uz]*v[0uz] - self[0uz]*v[2uz],
                     self[0uz]*v[1uz] - self[1uz]*v[0uz] };
        }
    };

    // Right-side operator overloads
    template <typename T, std::size_t N, typename T2, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE>
    constexpr auto operator*(const T& s, const VectorBase<T2, N, S2, OTHERSTORAGE> &v) { return v.map([&s](const T& e) { return e * s; }); }
    template <typename T, std::size_t N, typename T2, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE>
    constexpr auto operator/(const T& s, const VectorBase<T2, N, S2, OTHERSTORAGE> &v) { return v.map([&s](const T& e) { return e / s; }); }
    template <typename T, std::size_t N, std::ptrdiff_t S, STORAGECLASS STORAGETYPE>
    constexpr std::ostream& operator<<(std::ostream& os, const VectorBase<T, N, S, STORAGETYPE>& v) {
        for (std::size_t i = 0uz; i < N; ++i)
            os << (i ? " " : "") << v[i];
        return os;
    }


    // MATRIX
    template <typename T, std::size_t M, std::size_t N, std::ptrdiff_t S, STORAGECLASS STORAGETYPE>
    class MatrixBase : public STORAGETYPE<T, M, N, S>, public TensorRoot<T, M, N> {
    private:
        template <typename, std::size_t, std::size_t>
        friend class TensorRoot;

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
        // Constructors
        using STORAGETYPE<T, M, N, S>::STORAGETYPE;

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
        template<typename T2, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE>
        constexpr auto operator+(const MatrixBase<T2, M, N, S2, OTHERSTORAGE>& m) const { return this->binaryMapInternal([](const T& e1, const T2& e2){ return e1 + e2; }, m, MAKEINDICES(M*N)); }
        template<typename T2, std::ptrdiff_t S2, STORAGECLASS OTHERSTORAGE>
        constexpr auto operator-(const MatrixBase<T2, M, N, S2, OTHERSTORAGE>& m) const { return this->binaryMapInternal([](const T& e1, const T2& e2){ return e1 - e2; }, m, MAKEINDICES(M*N)); }
    };

    // Right-side operator overloads
    template <typename T, std::size_t M, std::size_t N, std::ptrdiff_t S, STORAGECLASS STORAGETYPE>
    constexpr std::ostream& operator<<(std::ostream& os, const MatrixBase<T, M, N, S, STORAGETYPE>& m) {
        for (std::size_t i = 0uz; i < M; ++i)
            for (std::size_t j = 0uz; j < N; ++j)
                os << (j ? " " : (i ? "\n" : "")) << m[i, j];
        return os;
    }


    ///////////////////
    // APPLIED TYPES //
    ///////////////////
    // These 4 are supposed to be the things you actually use; everything else above is inherited

    // Vector value-type struct
    template <typename T, std::size_t N>
    class Vector : public VectorBase<T, N, 1z, ValueType> {
        using VectorBase<T, N, 1z, ValueType>::VectorBase;
    };
    // Template deduction guide to automatically deduce N from initializer lists
    template <typename T, std::same_as<T>... Ts>
    Vector(T&&, Ts&&...) -> Vector<T, 1uz + sizeof...(Ts)>;

    // Vector reference-type struct
    template <typename T, std::size_t N, std::ptrdiff_t S = 1z>
    class VectorRef : public VectorBase<T, N, S, ReferenceType> {
        using VectorBase<T, N, S, ReferenceType>::VectorBase;
    };

    // Matrix value-type struct
    template <typename T, std::size_t M, std::size_t N>
    class Matrix : public MatrixBase<T, M, N, 1z, ValueType> {
    private:
        template <std::size_t... IDX>
        constexpr Matrix(T (&&payload)[M][N], std::index_sequence<IDX...>) : MatrixBase<T, M, N, 1z, ValueType>(std::forward<T>(payload[IDX/N][IDX%N])...) {}

    public:
        // Value-initialization constructor
        constexpr Matrix(T (&&payload)[M][N]) : Matrix(std::forward<T[M][N]>(payload), MAKEINDICES(M*N)) {}
    };

    // Matrix reference-type struct
    // TODO

    #undef COPYCONSTFORTYPE
    #undef STORAGECLASS
    #undef MAKEINDICES
}
#endif
