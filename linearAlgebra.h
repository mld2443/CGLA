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
#include <cstddef>     // size_t, ptrdiff_t
#include <iostream>    // ostream
#include <type_traits> // conditional_t, is_const_v, is_same_v, remove_reference_t
#include <utility>     // forward, index_sequence, make_index_sequence
#include <algorithm>   // min

// Helper macros to reduce clutter, undefined at end of file
#define COPYCONST(T1, ...) std::conditional_t<std::is_const_v<std::remove_reference_t<T1>>, const __VA_ARGS__, __VA_ARGS__>
#define MAKESEQUENCE(SIZE) std::make_index_sequence<SIZE>{}
#define SEQUENCE(NAME) std::index_sequence<NAME>&&

namespace util {
    template <std::size_t...> class TList {};

    constexpr auto minimum(const auto& first, const auto&... rest) {
        if constexpr (sizeof...(rest))
            return [&](this auto minimumImpl, const auto& first, const auto& second, const auto&... rest) constexpr {
                auto smaller = (first < second) ? first : second;
                if constexpr (sizeof...(rest)) return minimumImpl(smaller, rest...);
                else                           return smaller;
            }(first, rest...);
        else
            return first;
    }
}

namespace linalg {
    template <class StorageBase, typename T, std::size_t... DIMS>
    struct MultidimType;

    /////////////////////
    // STORAGE CLASSES //
    /////////////////////

    // Reference class, transient and transparent
    template <class ParentClass, std::size_t C, std::size_t... DIMSANDSTEPS>
    struct ReferenceClass {
    protected:
        static constexpr std::size_t COUNT = C;
        static constexpr bool ISREF = true;

        constexpr decltype(auto) get(this auto& self, std::size_t i) {
            return self.ref.get([]<std::size_t LASTSTEP, std::size_t DIM, std::size_t MYSTEP, std::size_t... REMAINS>(this auto translateIndex, std::size_t i, std::size_t accum) constexpr {
                constexpr std::size_t THISSTEP = LASTSTEP / DIM;
                if constexpr (sizeof...(REMAINS) / 2uz)
                    return translateIndex.template operator()<THISSTEP, REMAINS...>(i % THISSTEP, accum + MYSTEP * (i / THISSTEP));
                else
                    return accum + MYSTEP * (i / THISSTEP);
            }.template operator()<COUNT, DIMSANDSTEPS...>(i, self.offset));
        }

        template <class Self>
        constexpr decltype(auto) deref(this Self&& self, auto first, auto... inds) {
            return []<std::size_t THISDIM, std::size_t THISSTEP, std::size_t... RESTDIMSSTEPS, std::size_t... NEWDIMSANDSTEPS, std::size_t... NEWDIMS>(
                        this auto getSubstruct, Self&& self,
                        util::TList<NEWDIMSANDSTEPS...>&&, util::TList<NEWDIMS...>&&,
                        std::size_t offset, auto nextInd, auto... restInds) constexpr -> decltype(auto) {
                if constexpr (std::is_same_v<decltype(nextInd), char>) { // nextInd is a wildcard
                    if constexpr (sizeof...(restInds))             // more given indices after this wildcard
                        return getSubstruct.template operator()<RESTDIMSSTEPS...>(std::forward<Self>(self), util::TList<NEWDIMSANDSTEPS..., THISDIM, THISSTEP>{}, util::TList<NEWDIMS..., THISDIM>{}, offset, restInds...);
                    else if constexpr (sizeof...(RESTDIMSSTEPS))   // remaining dimensions are implied wildcards
                        return getSubstruct.template operator()<RESTDIMSSTEPS...>(std::forward<Self>(self), util::TList<NEWDIMSANDSTEPS..., THISDIM, THISSTEP>{}, util::TList<NEWDIMS..., THISDIM>{}, offset, '*');
                    else                                           // final index was given or implied wildcard
                        return MultidimType<ReferenceClass<ParentClass, (NEWDIMS * ... * THISDIM), NEWDIMSANDSTEPS..., THISDIM, 1uz>, COPYCONST(Self, std::remove_cvref_t<decltype(*self.ref.data)>), NEWDIMS..., THISDIM>(self.ref, offset);
                } else {
                    offset += THISSTEP * static_cast<std::size_t>(nextInd);
                    if constexpr (sizeof...(restInds))             // more constraints to get through and/or there are unconstrained dimensions
                        return getSubstruct.template operator()<RESTDIMSSTEPS...>(std::forward<Self>(self), util::TList<NEWDIMSANDSTEPS...>{}, util::TList<NEWDIMS...>{}, offset, restInds...);
                    else if constexpr (sizeof...(RESTDIMSSTEPS))   // remaining dimensions are implied wildcards
                        return getSubstruct.template operator()<RESTDIMSSTEPS...>(std::forward<Self>(self), util::TList<NEWDIMSANDSTEPS...>{}, util::TList<NEWDIMS...>{}, offset, '*');
                    else if constexpr (sizeof...(NEWDIMSANDSTEPS)) // all indices were given but at least one was a wildcard
                        return MultidimType<ReferenceClass<ParentClass, (NEWDIMS * ...), NEWDIMSANDSTEPS...>, COPYCONST(Self, std::remove_cvref_t<decltype(*self.ref.data)>), NEWDIMS...>(self.ref, offset);
                    else                                           // all indices given, no wildcards
                        return self.ref.get(offset);
                }
            }.template operator()<DIMSANDSTEPS...>(std::forward<Self>(self), {}, {}, self.offset, first, inds...);
        }

    public:
        constexpr ReferenceClass(ParentClass& ref, std::size_t offset) : ref(ref), offset(offset) {}

    protected:
        ParentClass& ref;
        std::size_t offset;
    };

    // Simple pointer type, takes user-supplied pointer and optional stride
    template <typename T, std::size_t C, std::ptrdiff_t STRIDE = 1z>
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
        constexpr RecursiveValueClass(SEQUENCE(IDX...), NestedArray* first) : data{ first[IDX]... } {}

    protected:
        constexpr RecursiveValueClass(NestedArray&& first) : RecursiveValueClass<T, COUNT>(MAKESEQUENCE(COUNT), &first) {}
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

    //////////////////////
    // Multidimensional //
    //////////////////////

    // Convenience aliases
    template <typename T, std::size_t... DIMS>
    using Multidimensional = MultidimType<ValueClass<T, DIMS...>, T, DIMS...>;
    template <typename T, std::size_t M, std::size_t N, class StorageType = ValueClass<T, M, N>>
    using Matrix = MultidimType<StorageType, T, M, N>;
    template <typename T, std::size_t M, std::size_t N, std::ptrdiff_t STRIDE = 1z>
    using MatrixPtr = MultidimType<PointerClass<T, M * N, STRIDE>, T, M, N>;
    template <typename T, std::size_t N, class StorageType = ValueClass<T, N>>
    using Vector = MultidimType<StorageType, T, N>;
    template <typename T, std::size_t N, std::ptrdiff_t STRIDE = 1z>
    using VectorPtr = MultidimType<PointerClass<T, N, STRIDE>, T, N>;

    // Concepts for dimension-dependant specializations
    template <class C> concept isVector = requires { C::order(); } && C::order() == 1uz;
    template <class C> concept isMatrix = requires { C::order(); } && C::order() == 2uz;
    template <class C> concept nonArray = !requires(C** x) { []<class S, class T, std::size_t... DIMS>(MultidimType<S, T, DIMS...>**){}(x); };
    template <class C, typename T, std::size_t... DIMS> concept isMultidim = requires(C** x) { []<class Storage>(MultidimType<Storage, T, DIMS...>**){}(x); };

    // Deduction guides for value-initialization
    // Anything higher than 10-dimensional can still be value-initialized, but template params must be explicit
    template <typename T, std::size_t D0>
    MultidimType(T (&&)[D0]) -> MultidimType<ValueClass<T, D0>, T, D0>;
    template <typename T, std::size_t D0, std::size_t D1>
    MultidimType(T (&&)[D0][D1]) -> MultidimType<ValueClass<T, D0, D1>, T, D0, D1>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2>
    MultidimType(T (&&)[D0][D1][D2]) -> MultidimType<ValueClass<T, D0, D1, D2>, T, D0, D1, D2>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3>
    MultidimType(T (&&)[D0][D1][D2][D3]) -> MultidimType<ValueClass<T, D0, D1, D2, D3>, T, D0, D1, D2, D3>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4>
    MultidimType(T (&&)[D0][D1][D2][D3][D4]) -> MultidimType<ValueClass<T, D0, D1, D2, D3, D4>, T, D0, D1, D2, D3, D4>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5>
    MultidimType(T (&&)[D0][D1][D2][D3][D4][D5]) -> MultidimType<ValueClass<T, D0, D1, D2, D3, D4, D5>, T, D0, D1, D2, D3, D4, D5>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6>
    MultidimType(T (&&)[D0][D1][D2][D3][D4][D5][D6]) -> MultidimType<ValueClass<T, D0, D1, D2, D3, D4, D5, D6>, T, D0, D1, D2, D3, D4, D5, D6>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7>
    MultidimType(T (&&)[D0][D1][D2][D3][D4][D5][D6][D7]) -> MultidimType<ValueClass<T, D0, D1, D2, D3, D4, D5, D6, D7>, T, D0, D1, D2, D3, D4, D5, D6, D7>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7, std::size_t D8>
    MultidimType(T (&&)[D0][D1][D2][D3][D4][D5][D6][D7][D8]) -> MultidimType<ValueClass<T, D0, D1, D2, D3, D4, D5, D6, D7, D8>, T, D0, D1, D2, D3, D4, D5, D6, D7, D8>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7, std::size_t D8, std::size_t D9>
    MultidimType(T (&&)[D0][D1][D2][D3][D4][D5][D6][D7][D8][D9]) -> MultidimType<ValueClass<T, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9>, T, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9>;

    template <class StorageClass, typename T, std::size_t... DIMS>
    struct MultidimType final : StorageClass {
    private:
        template <class, std::size_t, std::size_t...>
        friend struct ReferenceClass;    // Allows ReferenceType to use protected get()
        template <class, typename, std::size_t...>
        friend struct MultidimType;      // Allows different instantiations to use protected get()
        template <class OtherClass, typename T2, std::size_t FIRSTDIM, std::size_t... RESTDIMS>
        friend constexpr std::ostream& operator<<(std::ostream& os, const MultidimType<OtherClass, T2, FIRSTDIM, RESTDIMS...>& t);
        using StorageClass::COUNT;
        using StorageClass::get;         // Accessor addressing flat array of data, used internally to perform mappings and iterate

        // Special 'template container' prettyPrint() uses to build compile-time whitespace
        template <char... Cs> struct TString { static constexpr char STR[] = {Cs..., '\0'}; };

        // Helper for operator<<, displays arbitrary dimensional structures in a human-readable format
        template <std::size_t STEP, std::size_t THISDIM, std::size_t NEXTDIM = 0uz, std::size_t... RESTDIMS, char... PRFX, std::size_t... IDX>
        constexpr void prettyPrint(std::ostream& os, SEQUENCE(IDX...), std::size_t offset = 0uz, TString<PRFX...> prefix = {}) const {
            auto genSpace = []<std::size_t... IDX2>(SEQUENCE(IDX2...)) constexpr { return TString<PRFX..., (' ' + static_cast<char>(0uz & IDX2))...>(); };

            constexpr size_t DIMSREMAINING = sizeof...(RESTDIMS) + (NEXTDIM > 0uz) + 1uz;
            if constexpr (DIMSREMAINING > 3uz && DIMSREMAINING % 3uz != 0uz )
                os << (offset ? "\n" : "");

            if constexpr (DIMSREMAINING % 3uz == 0uz)
                (prettyPrint<STEP / NEXTDIM, NEXTDIM, RESTDIMS...>(os, MAKESEQUENCE(NEXTDIM), offset + IDX * STEP, genSpace(MAKESEQUENCE(((DIMSREMAINING - 3uz) ? (DIMSREMAINING - 3uz) : 3uz) * IDX))), ...);
            else if constexpr (NEXTDIM)
                (prettyPrint<STEP / NEXTDIM, NEXTDIM, RESTDIMS...>(os, MAKESEQUENCE(NEXTDIM), offset + IDX * STEP, TString<PRFX...>{}), ...);
            else {
                os << (offset ? "\n" : "") << prefix.STR;
                ((os << (IDX ? ", " : "") << get(offset + IDX)), ...) << ((offset + THISDIM < COUNT) ? "," : "");
            }
        }

    public:
        // Constructors
        using StorageClass::StorageClass;
        static constexpr auto broadcast(T&& s) { return [&]<std::size_t... IDX>(SEQUENCE(IDX...)) constexpr { return MultidimType<StorageClass, T, DIMS...>{ s + T(IDX & 0uz)... }; }(MAKESEQUENCE(COUNT)); }

        // Iterator for for-each loops
        template <class QualifiedType>
        struct Iterator {
        public:
            constexpr Iterator(QualifiedType& ref, std::size_t offset = 0uz) : ref(ref), pos(offset) {}

            constexpr decltype(auto) operator*() const { return ref.get(pos); }
            constexpr auto operator++() { ++pos; return *this; }
            constexpr bool operator==(const Iterator& o) const { return pos == o.pos; }

        private:
            QualifiedType& ref;
            std::size_t pos;
        };
        constexpr auto begin(this auto& self) { return Iterator<decltype(self)>{ self }; }
        constexpr auto   end(this auto& self) { return Iterator<decltype(self)>{ self , COUNT }; }

        // Metadata
        static constexpr std::size_t count() { return COUNT; }
        static constexpr std::size_t order() { return sizeof...(DIMS); }

        // Functional programming
        constexpr auto map(auto&& func) const {
            return [this]<std::size_t... IDX>(auto& func, SEQUENCE(IDX...)) constexpr {
                return Multidimensional<decltype(func(T())), DIMS...>{ func(get(IDX))... };
            }(func, MAKESEQUENCE(COUNT));
        }
        constexpr auto binaryMap(auto&& func, const auto& t) const {
            return [this]<class OtherClass, typename T2, std::size_t... IDX>(auto& func, const MultidimType<OtherClass, T2, DIMS...>& t, SEQUENCE(IDX...)) constexpr {
                return Multidimensional<decltype(func(T(), T2())), DIMS...>{ func(get(IDX), t.get(IDX))... };
            }(func, t, MAKESEQUENCE(COUNT));
        }
        inline void mapWrite(auto&& func) {
            [this]<std::size_t... IDX>(auto& func, SEQUENCE(IDX...)) constexpr {
                (func(get(IDX)), ...);
            }(func, MAKESEQUENCE(COUNT));
        }
        inline void binaryMapWrite(auto&& func, const auto& t) {
            [this]<class OtherClass, typename T2, std::size_t... IDX>(auto& func, const MultidimType<OtherClass, T2, DIMS...>& t, SEQUENCE(IDX...)) constexpr {
                (func(get(IDX), t.get(IDX)), ...);
            }(func, t, MAKESEQUENCE(COUNT));
        }
        constexpr auto reduce(auto&& func, auto starting) const {
            return [this]<std::size_t... IDX>(auto& func, auto starting, SEQUENCE(IDX...)) constexpr {
                return ((starting = func(starting, get(IDX))), ...);
            }(func, starting, MAKESEQUENCE(COUNT));
        }
        constexpr auto reduce(auto&& func) const {
            if constexpr (COUNT == 1uz)
                return get(0uz);
            else
                return [this]<std::size_t... IDX>(auto& func, T starting, SEQUENCE(IDX...)) constexpr {
                    return ((starting = func(starting, get(1uz + IDX))), ...);
                }(func, get(0uz), MAKESEQUENCE(COUNT - 1uz));
        }

        // Member operator overloads
        constexpr auto operator-(                      ) const { return map([  ](const T& e){ return    -e; }); }
        constexpr auto operator*(const nonArray auto& s) const { return map([&s](const T& e){ return e * s; }); }
        constexpr auto operator/(const nonArray auto& s) const { return map([&s](const T& e){ return e / s; }); }
        constexpr auto operator+(const          auto& t) const { return binaryMap([](const T& e1, const auto& e2){ return e1 + e2; }, t); }
        constexpr auto operator-(const          auto& t) const { return binaryMap([](const T& e1, const auto& e2){ return e1 - e2; }, t); }

        // Mutating operators
        inline auto& operator*=(const nonArray auto& s) { mapWrite([&s](T& e){ e *= s; }); return *this; }
        inline auto& operator/=(const nonArray auto& s) { mapWrite([&s](T& e){ e /= s; }); return *this; }
        inline auto& operator+=(const          auto& t) { binaryMapWrite([](T& e1, const auto& e2){ e1 += e2; }, t); return *this; }
        inline auto& operator-=(const          auto& t) { binaryMapWrite([](T& e1, const auto& e2){ e1 -= e2; }, t); return *this; }
        inline auto& operator= (const          auto& t) { binaryMapWrite([](T& e1, const auto& e2){ e1 =  e2; }, t); return *this; }

        template <std::size_t CONTRACTIONS, class OtherClass, typename T2, std::size_t... DIMS2> requires(CONTRACTIONS > 0uz)
        constexpr auto contract(this const MultidimType<StorageClass, T, DIMS...>& , const MultidimType<OtherClass, T2, DIMS2...>& ) requires([](std::size_t (&&d1)[sizeof...(DIMS)], std::size_t (&&d2)[sizeof...(DIMS2)]){
            for (std::size_t i = 0uz; i < CONTRACTIONS; ++i)
                if (d1[sizeof...(DIMS) - 1uz - i] != d2[i])
                    return false;
            return true;
        }({ DIMS... }, { DIMS2... })) {
            return []<std::size_t D1_FIRST, std::size_t... D1_REST, std::size_t D2_FIRST, std::size_t... D2_REST, std::size_t... FRONTDIMS>(this auto buildDims,
                        util::TList<D1_FIRST, D1_REST...>&&, util::TList<D2_FIRST, D2_REST...>&&, util::TList<FRONTDIMS...>&&) constexpr {
                if constexpr (sizeof...(D1_REST) >= CONTRACTIONS)                            // Accept dimensions off front of DIMS until only contracted dims remain
                    return buildDims(util::TList<D1_REST...>{}, util::TList<D2_FIRST, D2_REST...>{}, util::TList<FRONTDIMS..., D1_FIRST>{});
                else if constexpr (sizeof...(D2_REST) + CONTRACTIONS > sizeof...(DIMS2))    // Drop dimensions off front of DIMS2 until all contracted dims are gone
                    return buildDims(util::TList<D1_FIRST, D1_REST...>{}, util::TList<D2_REST...>{}, util::TList<FRONTDIMS...>{});
                else {
                    using ReturnType = Multidimensional<decltype(T() * T2()), FRONTDIMS..., D2_REST...>;

                    return ReturnType{};
                }
            }(util::TList<DIMS...>{}, util::TList<DIMS2...>{}, {});
        }

        // template<class OtherClass, typename T2, std::size_t... DIMS2> requires([](std::size_t (&&d1)[sizeof...(DIMS)], std::size_t (&&d2)[sizeof...(DIMS2)]){
        //     return d1[sizeof...(DIMS) - 1uz] == d2[0uz];
        // }({ DIMS... }, { DIMS2... }))
        // constexpr auto operator*(this const MultidimType<StorageType, T, DIMS...>& self, const MultidimType<OtherClass, T2, DIMS2...>& t) {
        //     return self.contract<1uz>(t);
        // }

        ///////////////
        // ACCESSORS //
        ///////////////

        template <class Self>
        constexpr decltype(auto) operator[](this Self&& self, auto first, auto... inds) requires (sizeof...(inds) < sizeof...(DIMS)) {
            if constexpr (std::remove_cvref_t<Self>::ISREF)
                return self.deref(first, inds...);
            else
                return []<std::size_t STEP, std::size_t THISDIM, std::size_t... RESTDIMS, std::size_t... DIMSANDSTEPS, std::size_t... NEWDIMS>(
                            this auto getSubstruct, Self&& self,
                            util::TList<DIMSANDSTEPS...>&&, util::TList<NEWDIMS...>&&,
                            std::size_t offset, auto nextInd, auto... restInds) constexpr -> decltype(auto) {
                    constexpr std::size_t THISSTEP = STEP / THISDIM;
                    if constexpr (std::is_same_v<decltype(nextInd), char>) { // nextInd is a wildcard
                        if constexpr (sizeof...(restInds))          // more given indices after this wildcard
                            return getSubstruct.template operator()<THISSTEP, RESTDIMS...>(std::forward<Self>(self), util::TList<DIMSANDSTEPS..., THISDIM, THISSTEP>{}, util::TList<NEWDIMS..., THISDIM>{}, offset, restInds...);
                        else if constexpr (sizeof...(RESTDIMS))     // remaining dimensions are implied wildcards
                            return getSubstruct.template operator()<THISSTEP, RESTDIMS...>(std::forward<Self>(self), util::TList<DIMSANDSTEPS..., THISDIM, THISSTEP>{}, util::TList<NEWDIMS..., THISDIM>{}, offset, '*');
                        else                                        // final index was given as or implied to be a wildcard
                            return MultidimType<ReferenceClass<Self, (NEWDIMS * ... * THISDIM), DIMSANDSTEPS..., THISDIM, 1uz>, COPYCONST(Self, T), NEWDIMS..., THISDIM>(std::forward<Self>(self), offset);
                    } else {
                        offset += THISSTEP * static_cast<std::size_t>(nextInd);
                        if constexpr (sizeof...(restInds))          // more constraints to get through and/or there are unconstrained dimensions
                            return getSubstruct.template operator()<THISSTEP, RESTDIMS...>(std::forward<Self>(self), util::TList<DIMSANDSTEPS...>{}, util::TList<NEWDIMS...>{}, offset, restInds...);
                        else if constexpr (sizeof...(RESTDIMS))     // remaining dimensions are implied wildcards
                            return getSubstruct.template operator()<THISSTEP, RESTDIMS...>(std::forward<Self>(self), util::TList<DIMSANDSTEPS...>{}, util::TList<NEWDIMS...>{}, offset, '*');
                        else if constexpr (sizeof...(DIMSANDSTEPS)) // all indices were given but at least one was a wildcard
                            return MultidimType<ReferenceClass<Self, (NEWDIMS * ...), DIMSANDSTEPS...>, COPYCONST(Self, T), NEWDIMS...>(std::forward<Self>(self), offset);
                        else                                        // all indices given, no wildcards
                            return self.get(offset);
                    }
                }.template operator()<COUNT, DIMS...>(std::forward<Self>(self), {}, {}, 0uz, first, inds...);
        }

        template <class Self>
        constexpr decltype(auto) getDiagonal(this Self& self) requires(std::remove_cvref_t<Self>::order() > 1uz) {
            constexpr std::size_t SMALLEST = []<std::size_t D0, std::size_t D1, std::size_t... REST>(this auto minimum) constexpr {
                if constexpr (sizeof...(REST)) return minimum.template operator()<std::min(D0, D1), REST...>();
                else                           return std::min(D0, D1);
            }.template operator()<DIMS...>();
            constexpr std::size_t STRIDE = []<std::size_t PRODUCT, std::size_t, std::size_t... REST>(this auto calcStride) constexpr {
                if constexpr (sizeof...(REST)) return calcStride.template operator()<PRODUCT + (REST * ...), REST...>();
                else                           return PRODUCT;
            }.template operator()<1uz, DIMS...>();

            return MultidimType<ReferenceClass<Self, SMALLEST, SMALLEST, STRIDE>, COPYCONST(Self, T), SMALLEST>{ self, 0uz };
        }

        ////////////////////////////
        // VECTOR SPECIALIZATIONS //
        ////////////////////////////

        constexpr auto dot(this const isVector auto& self, const isVector auto& v) requires(COUNT == std::remove_cvref_t<decltype(v)>::COUNT) {
            return self.binaryMap([](const auto& a, const auto& b){ return a * b; }, v).reduce([](const auto& a, const auto& b){ return a + b; });
        }
        constexpr T magnitudeSqr(this const isVector auto& self) { return self.dot(self);                 }
        constexpr T    magnitude(this const isVector auto& self) { return std::sqrt(self.magnitudeSqr()); }
        constexpr auto direction(this const isVector auto& self) { return self / self.magnitude();        }
        //constexpr auto covector(this isVector auto& self) { return MultidimType<ReferenceClass<std::remove_reference_t<decltype(self)>, COUNT, COUNT, 1uz, 1uz, 1uz>, T, DIMS..., 1uz>{ self, 0uz }; }
        template <isVector Self> constexpr Matrix<T, COUNT, 1uz, ReferenceClass<Self, COUNT, COUNT, 1uz, 1uz, 1uz>> covector(this Self& self) { return { self, 0uz }; }
        template <isVector Self> constexpr operator Matrix<T, 1uz, COUNT, ReferenceClass<Self, COUNT, 1uz, 1uz, COUNT, 1uz>>(this Self& self) { return { self, 0uz }; }

        template <typename T2, class OtherClass>
        constexpr Vector<decltype(T()*T2()), 3uz> cross(this const Vector<T, 3uz, StorageClass>& self, const Vector<T2, 3uz, OtherClass>& v) {
            return { self[1uz]*v[2uz] - self[2uz]*v[1uz],
                     self[2uz]*v[0uz] - self[0uz]*v[2uz],
                     self[0uz]*v[1uz] - self[1uz]*v[0uz] };
        }

        ////////////////////////////
        // MATRIX SPECIALIZATIONS //
        ////////////////////////////

        static constexpr auto Identity() requires(order() == 2uz) {
            return []<std::size_t M, std::size_t N, std::size_t... IDX>(SEQUENCE(IDX...)) constexpr requires(M == N) {
                return Matrix<T, M, N>{ T((IDX % (M + 1uz)) == 0uz)... };
            }.template operator()<DIMS...>(MAKESEQUENCE(COUNT));
        }

        // Matrix specific accessors
        constexpr decltype(auto) getRow(this isMatrix auto& self, std::size_t r) { return self[r]; }
        constexpr decltype(auto) getCol(this isMatrix auto& self, std::size_t c) { return self['*', c]; }

        template <std::size_t N> requires(N > 1uz)
        constexpr T determinant(this const Matrix<T, N, N, StorageClass>&) {
            return T(); //FIXME
        }

        template<typename T2, std::size_t M, std::size_t N, std::size_t O, class OtherClass>
        constexpr auto operator*(this const Matrix<T, M, N, StorageClass>& self, const Matrix<T2, N, O, OtherClass>& m) {
            return []<std::size_t... IDX>(const auto& m1, const auto& m2, SEQUENCE(IDX...)) constexpr {
                return Matrix<decltype(T()*T2()), M, O>{ m1.getRow(IDX / O).dot(m2.getCol(IDX % O))... };
            }(self, m, MAKESEQUENCE(M*O));
        }
    };

    // Right-side operator overloads
    template <class StorageType, typename T, std::size_t... DIMS>
    constexpr auto operator*(const nonArray auto& s, const MultidimType<StorageType, T, DIMS...> &t) { return t.map([&s](const T& e) { return s * e; }); }
    template <class StorageType, typename T, std::size_t... DIMS>
    constexpr auto operator/(const nonArray auto& s, const MultidimType<StorageType, T, DIMS...> &t) { return t.map([&s](const T& e) { return s / e; }); }
    template <class StorageType, typename T, std::size_t FIRSTDIM, std::size_t... RESTDIMS>
    constexpr std::ostream& operator<<(std::ostream& os, const MultidimType<StorageType, T, FIRSTDIM, RESTDIMS...>& t) {
        t.template prettyPrint<(RESTDIMS * ... * 1uz), FIRSTDIM, RESTDIMS...>(os, MAKESEQUENCE(FIRSTDIM));
        return os;
    }
}

#undef COPYCONST
#undef MAKESEQUENCE
#undef SEQUENCE
