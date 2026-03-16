module;

#include <cmath>       // sqrt
#include <cstddef>     // ptrdiff_t, size_t
#include <iostream>    // ostream
#include <type_traits> // remove_cvref_t
#include <utility>     // forward

export module linalg;

import util;
import meta;

export import :reference;
export import :pointer;
export import :value;


namespace linalg {
    template <class STORAGEBASE, typename T, std::size_t... DIMS>
    struct TensorType;

    // Convenience aliases
    export template <typename T, std::size_t... DIMS>
    using Tensor = TensorType<ValueClass<T, DIMS...>, T, DIMS...>;
    export template <typename T, std::size_t M, std::size_t N, class StorageType = ValueClass<T, M, N>>
    using Matrix = TensorType<StorageType, T, M, N>;
    export template <typename T, std::size_t M, std::size_t N, std::ptrdiff_t STRIDE = 1z>
    using MatrixPtr = TensorType<PointerClass<T, M * N, STRIDE>, T, M, N>;
    export template <typename T, std::size_t N, class StorageType = ValueClass<T, N>>
    using Vector = TensorType<StorageType, T, N>;
    export template <typename T, std::size_t N, std::ptrdiff_t STRIDE = 1z>
    using VectorPtr = TensorType<PointerClass<T, N, STRIDE>, T, N>;

    // Concepts for dimension-dependant specializations
    template <class C> concept isVector = requires { C::order(); } && C::order() == 1uz;
    template <class C> concept isMatrix = requires { C::order(); } && C::order() == 2uz;
    template <class C> concept nonArray = !requires(C** x) { []<class S, class T, std::size_t... DIMS>(TensorType<S, T, DIMS...>**){}(x); };
    template <class C, typename T, std::size_t... DIMS> concept isMultidim = requires(C** x) { []<class Storage>(TensorType<Storage, T, DIMS...>**){}(x); };

    // Deduction guides for value-initialization
    // Anything higher than 10-dimensional can still be value-initialized, but template params must be explicit
    template <typename T, std::size_t D0>
    TensorType(T (&&)[D0]) -> TensorType<ValueClass<T, D0>, T, D0>;
    template <typename T, std::size_t D0, std::size_t D1>
    TensorType(T (&&)[D0][D1]) -> TensorType<ValueClass<T, D0, D1>, T, D0, D1>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2>
    TensorType(T (&&)[D0][D1][D2]) -> TensorType<ValueClass<T, D0, D1, D2>, T, D0, D1, D2>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3>
    TensorType(T (&&)[D0][D1][D2][D3]) -> TensorType<ValueClass<T, D0, D1, D2, D3>, T, D0, D1, D2, D3>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4>
    TensorType(T (&&)[D0][D1][D2][D3][D4]) -> TensorType<ValueClass<T, D0, D1, D2, D3, D4>, T, D0, D1, D2, D3, D4>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5>
    TensorType(T (&&)[D0][D1][D2][D3][D4][D5]) -> TensorType<ValueClass<T, D0, D1, D2, D3, D4, D5>, T, D0, D1, D2, D3, D4, D5>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6>
    TensorType(T (&&)[D0][D1][D2][D3][D4][D5][D6]) -> TensorType<ValueClass<T, D0, D1, D2, D3, D4, D5, D6>, T, D0, D1, D2, D3, D4, D5, D6>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7>
    TensorType(T (&&)[D0][D1][D2][D3][D4][D5][D6][D7]) -> TensorType<ValueClass<T, D0, D1, D2, D3, D4, D5, D6, D7>, T, D0, D1, D2, D3, D4, D5, D6, D7>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7, std::size_t D8>
    TensorType(T (&&)[D0][D1][D2][D3][D4][D5][D6][D7][D8]) -> TensorType<ValueClass<T, D0, D1, D2, D3, D4, D5, D6, D7, D8>, T, D0, D1, D2, D3, D4, D5, D6, D7, D8>;
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, std::size_t D4, std::size_t D5, std::size_t D6, std::size_t D7, std::size_t D8, std::size_t D9>
    TensorType(T (&&)[D0][D1][D2][D3][D4][D5][D6][D7][D8][D9]) -> TensorType<ValueClass<T, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9>, T, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9>;

    template <class STORAGECLASS, typename T, std::size_t... DIMS>
    struct TensorType final : STORAGECLASS {
    private:
        template <class, std::size_t, std::size_t...>
        friend struct ReferenceClass; // Allows ReferenceType to use protected get()
        template <class, typename, std::size_t...>
        friend struct TensorType;     // Allows different instantiations to use protected get()
        template <class OTHERCLASS, typename T2, std::size_t FIRSTDIM, std::size_t... RESTDIMS>
        friend constexpr std::ostream& operator<<(std::ostream& os, const TensorType<OTHERCLASS, T2, FIRSTDIM, RESTDIMS...>& t);
        using STORAGECLASS::COUNT;
        using STORAGECLASS::get;      // Accessor addressing flat array of data, used internally to perform mappings and iterate

        // Helper for operator<<, displays arbitrary dimensional structures in a human-readable format
        template <std::size_t STEP, std::size_t THISDIM, std::size_t NEXTDIM = 0uz, std::size_t... RESTDIMS, char... STR, std::size_t... IDX>
        constexpr void prettyPrint(std::ostream& os, meta::List<IDX...>&&, std::size_t offset = 0uz, meta::String<STR...>&& prefix = {}) const {
            constexpr std::size_t DIMSREMAINING = sizeof...(RESTDIMS) + (NEXTDIM > 0uz) + 1uz;
            if constexpr (DIMSREMAINING > 3uz && DIMSREMAINING % 3uz != 0uz )
                os << (offset ? "\n" : "");

            if constexpr (DIMSREMAINING % 3uz == 0uz)
                (prettyPrint<STEP / NEXTDIM, NEXTDIM, RESTDIMS...>(
                    os,
                    meta::sequenceList<NEXTDIM>(),
                    offset + IDX * STEP,
                    meta::repeatedList<(DIMSREMAINING > 3uz ? DIMSREMAINING - 3uz : 3uz) * IDX, ' ', meta::String>()
                ), ...);
            else if constexpr (NEXTDIM)
                (prettyPrint<STEP / NEXTDIM, NEXTDIM, RESTDIMS...>(
                    os,
                    meta::sequenceList<NEXTDIM>(),
                    offset + IDX * STEP,
                    meta::String<STR...>{}
                ), ...);
            else {
                os << (offset ? "\n" : "") << prefix.STR;
                ((os << (IDX ? ", " : "") << get(offset + IDX)), ...) << ((offset + THISDIM < COUNT) ? "," : "");
            }
        }

    public:
        // Constructors
        using STORAGECLASS::STORAGECLASS;
        static constexpr auto broadcast(T&& s) { return [&]<std::size_t... IDX>(meta::List<IDX...>&&) constexpr {
            return TensorType<STORAGECLASS, T, DIMS...>{ ((void)IDX, s)... };
        }(meta::sequenceList<COUNT>()); }

        // Iterator for for-each loops
        template <class QUALIFIEDTYPE>
        struct FwdIterator {
        public:
            constexpr FwdIterator(QUALIFIEDTYPE& ref, std::size_t offset = 0uz) : ref(ref), pos(offset) {}

            constexpr decltype(auto) operator*() const { return ref.get(pos); }
            constexpr auto operator++() { ++pos; return *this; }
            constexpr bool operator==(const FwdIterator& o) const { return pos == o.pos; }

        private:
            QUALIFIEDTYPE& ref;
            std::size_t pos;
        };
        constexpr auto begin(this auto& self) { return FwdIterator<decltype(self)>{ self }; }
        constexpr auto   end(this auto& self) { return FwdIterator<decltype(self)>{ self , COUNT }; }

        // Metadata
        static constexpr std::size_t count() { return COUNT; }
        static constexpr std::size_t order() { return sizeof...(DIMS); }
        static constexpr auto shape() { return Vector{ { DIMS... } }; }

        // Functional programming
        constexpr auto map(auto&& func) const {
            return [this]<std::size_t... IDX>(auto& func, meta::List<IDX...>&&) constexpr {
                return Tensor<decltype(func(T())), DIMS...>{ func(get(IDX))... };
            }(func, meta::sequenceList<COUNT>());
        }
        constexpr auto binaryMap(auto&& func, const auto& t) const {
            return [this]<class OTHERCLASS, typename T2, std::size_t... IDX>(auto& func, const TensorType<OTHERCLASS, T2, DIMS...>& t, meta::List<IDX...>&&) constexpr {
                return Tensor<decltype(func(T(), T2())), DIMS...>{ func(get(IDX), t.get(IDX))... };
            }(func, t, meta::sequenceList<COUNT>());
        }
        inline void mapWrite(auto&& func) {
            [this]<std::size_t... IDX>(auto& func, meta::List<IDX...>&&) constexpr {
                (func(get(IDX)), ...);
            }(func, meta::sequenceList<COUNT>());
        }
        inline void binaryMapWrite(auto&& func, const auto& t) {
            [this]<class OTHERCLASS, typename T2, std::size_t... IDX>(auto& func, const TensorType<OTHERCLASS, T2, DIMS...>& t, meta::List<IDX...>&&) constexpr {
                (func(get(IDX), t.get(IDX)), ...);
            }(func, t, meta::sequenceList<COUNT>());
        }
        constexpr auto reduce(auto&& func, auto starting) const {
            return [this]<std::size_t... IDX>(auto& func, auto starting, meta::List<IDX...>&&) constexpr {
                return ((starting = func(starting, get(IDX))), ...);
            }(func, starting, meta::sequenceList<COUNT>());
        }
        constexpr auto reduce(auto&& func) const {
            if constexpr (COUNT == 1uz)
                return get(0uz);
            else
                return [this]<std::size_t... IDX>(auto& func, T starting, meta::List<IDX...>&&) constexpr {
                    return ((starting = func(starting, get(1uz + IDX))), ...);
                }(func, get(0uz), meta::sequenceList<COUNT - 1uz>());
        }

        // Member operator overloads
        constexpr auto operator-(                      ) const { return map([  ](const T& e) constexpr { return    -e; }); }
        constexpr auto operator*(const nonArray auto& s) const { return map([&s](const T& e) constexpr { return e * s; }); }
        constexpr auto operator/(const nonArray auto& s) const { return map([&s](const T& e) constexpr { return e / s; }); }
        constexpr auto operator+(const          auto& t) const { return binaryMap([](const T& e1, const auto& e2) constexpr { return e1 + e2; }, t); }
        constexpr auto operator-(const          auto& t) const { return binaryMap([](const T& e1, const auto& e2) constexpr { return e1 - e2; }, t); }

        // Mutating operators
        inline auto& operator*=(const nonArray auto& s) { mapWrite([&s](T& e) constexpr { e *= s; }); return *this; }
        inline auto& operator/=(const nonArray auto& s) { mapWrite([&s](T& e) constexpr { e /= s; }); return *this; }
        inline auto& operator+=(const          auto& t) { binaryMapWrite([](T& e1, const auto& e2) constexpr { e1 += e2; }, t); return *this; }
        inline auto& operator-=(const          auto& t) { binaryMapWrite([](T& e1, const auto& e2) constexpr { e1 -= e2; }, t); return *this; }
        inline auto& operator= (const          auto& t) { binaryMapWrite([](T& e1, const auto& e2) constexpr { e1 =  e2; }, t); return *this; }

        template <std::size_t CONTRACTIONS, class OTHERCLASS, typename T2, std::size_t... DIMS2> requires(
            CONTRACTIONS > 0uz &&
            CONTRACTIONS <= sizeof...(DIMS) &&
            CONTRACTIONS <= sizeof...(DIMS2))
        constexpr auto contract(this const TensorType<STORAGECLASS, T, DIMS...>&, const TensorType<OTHERCLASS, T2, DIMS2...>&)
                //requires(std::ranges::equal(std::views::drop(lhs.shape(), lhs.order() - CONTRACTIONS), std::views::take(rhs.shape(), CONTRACTIONS)))
        {
            return []<std::size_t ROTATIONS, std::size_t D1_FIRST, std::size_t... D1_REST, std::size_t D2_FIRST, std::size_t... D2_REST>(
                        this auto buildDims, meta::List<D1_FIRST, D1_REST...>&&, meta::List<D2_FIRST, D2_REST...>&&) constexpr {
                if constexpr (ROTATIONS > 0uz) {                                             // Rotate first list until contracted dimensions are at the front
                    return buildDims.template operator()<ROTATIONS - 1uz>(meta::List<D1_REST..., D1_FIRST>{}, meta::List<D2_FIRST, D2_REST...>{});
                } else if constexpr (sizeof...(D2_REST) > sizeof...(DIMS2) - CONTRACTIONS) { // Drop front of both lists until all contracted dimensions are gone
                    static_assert(D1_FIRST == D2_FIRST, "Contracted dimensions do not match!");
                    return buildDims.template operator()<0uz>(meta::List<D1_REST...>{}, meta::List<D2_REST...>{});
                } else {
                    using ReturnType = Tensor<decltype(T() * T2()), D1_REST..., D2_REST...>;

                    return ReturnType{};
                }
            }.template operator()<sizeof...(DIMS) - CONTRACTIONS>(meta::List<DIMS...>{}, meta::List<DIMS2...>{});
        }

        // template<class OTHERCLASS, typename T2, std::size_t... DIMS2> requires([](std::size_t (&&d1)[sizeof...(DIMS)], std::size_t (&&d2)[sizeof...(DIMS2)]){
        //     return d1[sizeof...(DIMS) - 1uz] == d2[0uz];
        // }({ DIMS... }, { DIMS2... }))
        // constexpr auto operator*(this const MultidimType<STORAGETYPE, T, DIMS...>& self, const MultidimType<OTHERCLASS, T2, DIMS2...>& t) {
        //     return self.contract<1uz>(t);
        // }

        ///////////////
        // ACCESSORS //
        ///////////////

        template <class SELF>
        constexpr decltype(auto) operator[](this SELF&& self, auto first, auto... inds) requires (sizeof...(inds) < sizeof...(DIMS)) {
            if constexpr (std::remove_cvref_t<SELF>::ISREF)
                return self.deref(first, inds...);
            else
                return []<std::size_t STEP, std::size_t THISDIM, std::size_t... RESTDIMS, std::size_t... DIMSANDSTEPS, std::size_t... NEWDIMS>(
                            this auto getSubstruct, SELF&& self,
                            meta::List<DIMSANDSTEPS...>&&, meta::List<NEWDIMS...>&&,
                            std::size_t offset, auto nextInd, auto... restInds) constexpr -> decltype(auto) {
                    constexpr std::size_t THISSTEP = STEP / THISDIM;
                    if constexpr (std::is_same_v<decltype(nextInd), char>) { // nextInd is a wildcard
                        if constexpr (sizeof...(restInds))          // more given indices after this wildcard
                            return getSubstruct.template operator()<THISSTEP, RESTDIMS...>(std::forward<SELF>(self), meta::List<DIMSANDSTEPS..., THISDIM, THISSTEP>{}, meta::List<NEWDIMS..., THISDIM>{}, offset, restInds...);
                        else if constexpr (sizeof...(RESTDIMS))     // remaining dimensions are implied wildcards
                            return getSubstruct.template operator()<THISSTEP, RESTDIMS...>(std::forward<SELF>(self), meta::List<DIMSANDSTEPS..., THISDIM, THISSTEP>{}, meta::List<NEWDIMS..., THISDIM>{}, offset, '*');
                        else                                        // final index was given as or implied to be a wildcard
                            return TensorType<ReferenceClass<SELF, (NEWDIMS * ... * THISDIM), DIMSANDSTEPS..., THISDIM, 1uz>, meta::copyConst<SELF, T>, NEWDIMS..., THISDIM>(std::forward<SELF>(self), offset);
                    } else {
                        offset += THISSTEP * static_cast<std::size_t>(nextInd);
                        if constexpr (sizeof...(restInds))          // more constraints to get through and/or there are unconstrained dimensions
                            return getSubstruct.template operator()<THISSTEP, RESTDIMS...>(std::forward<SELF>(self), meta::List<DIMSANDSTEPS...>{}, meta::List<NEWDIMS...>{}, offset, restInds...);
                        else if constexpr (sizeof...(RESTDIMS))     // remaining dimensions are implied wildcards
                            return getSubstruct.template operator()<THISSTEP, RESTDIMS...>(std::forward<SELF>(self), meta::List<DIMSANDSTEPS...>{}, meta::List<NEWDIMS...>{}, offset, '*');
                        else if constexpr (sizeof...(DIMSANDSTEPS)) // all indices were given but at least one was a wildcard
                            return TensorType<ReferenceClass<SELF, (NEWDIMS * ...), DIMSANDSTEPS...>, meta::copyConst<SELF, T>, NEWDIMS...>(std::forward<SELF>(self), offset);
                        else                                        // all indices given, no wildcards
                            return self.get(offset);
                    }
                }.template operator()<COUNT, DIMS...>(std::forward<SELF>(self), {}, {}, 0uz, first, inds...);
        }

        template <class SELF>
        constexpr decltype(auto) getDiagonal(this SELF& self) requires(std::remove_cvref_t<SELF>::order() > 1uz) {
            constexpr std::size_t SMALLEST = util::minimum(DIMS...);
            constexpr std::size_t STRIDE = []<std::size_t PRODUCT, std::size_t, std::size_t... REST>(this auto calcStride) constexpr {
                if constexpr (sizeof...(REST)) return calcStride.template operator()<PRODUCT + (REST * ...), REST...>();
                else                           return PRODUCT;
            }.template operator()<1uz, DIMS...>();

            return TensorType<ReferenceClass<SELF, SMALLEST, SMALLEST, STRIDE>, meta::copyConst<SELF, T>, SMALLEST>{ self, 0uz };
        }

        ////////////////////////////
        // VECTOR SPECIALIZATIONS //
        ////////////////////////////

        constexpr auto dot(this const isVector auto& self, const isVector auto& v) requires(COUNT == std::remove_cvref_t<decltype(v)>::COUNT) {
            return self.binaryMap([](const auto& a, const auto& b) constexpr { return a * b; }, v).reduce([](const auto& a, const auto& b) constexpr { return a + b; });
        }
        constexpr T magnitudeSqr(this const isVector auto& self) { return self.dot(self);                 }
        constexpr T    magnitude(this const isVector auto& self) { return std::sqrt(self.magnitudeSqr()); }
        constexpr auto direction(this const isVector auto& self) { return self / self.magnitude();        }
        //constexpr auto covector(this isVector auto& self) { return MultidimType<ReferenceClass<std::remove_reference_t<decltype(self)>, COUNT, COUNT, 1uz, 1uz, 1uz>, T, DIMS..., 1uz>{ self, 0uz }; }
        template <isVector SELF> constexpr Matrix<T, COUNT, 1uz, ReferenceClass<SELF, COUNT, COUNT, 1uz, 1uz, 1uz>> covector(this SELF& self) { return { self, 0uz }; }
        template <isVector SELF> constexpr operator Matrix<T, 1uz, COUNT, ReferenceClass<SELF, COUNT, 1uz, 1uz, COUNT, 1uz>>(this SELF& self) { return { self, 0uz }; }

        template <typename T2, class OTHERCLASS>
        constexpr Vector<decltype(T()*T2()), 3uz> cross(this const Vector<T, 3uz, STORAGECLASS>& self, const Vector<T2, 3uz, OTHERCLASS>& v) {
            return { self[1uz]*v[2uz] - self[2uz]*v[1uz],
                     self[2uz]*v[0uz] - self[0uz]*v[2uz],
                     self[0uz]*v[1uz] - self[1uz]*v[0uz] };
        }

        ////////////////////////////
        // MATRIX SPECIALIZATIONS //
        ////////////////////////////

        static constexpr auto Identity() requires(order() == 2uz) {
            return []<std::size_t M, std::size_t N, std::size_t... IDX>(meta::List<IDX...>&&) constexpr requires(M == N) {
                return Matrix<T, M, N>{ T((IDX % (M + 1uz)) == 0uz)... };
            }.template operator()<DIMS...>(meta::sequenceList<COUNT>());
        }

        // Matrix specific accessors
        constexpr decltype(auto) getRow(this isMatrix auto& self, std::size_t r) { return self[r]; }
        constexpr decltype(auto) getCol(this isMatrix auto& self, std::size_t c) { return self['*', c]; }

        template <std::size_t N> requires(N > 1uz)
        constexpr T determinant(this const Matrix<T, N, N, STORAGECLASS>&) {
            return T(); //FIXME
        }

        template<typename T2, std::size_t M, std::size_t N, std::size_t O, class OTHERCLASS>
        constexpr auto operator*(this const Matrix<T, M, N, STORAGECLASS>& self, const Matrix<T2, N, O, OTHERCLASS>& m) {
            return []<std::size_t... IDX>(const auto& m1, const auto& m2, meta::List<IDX...>&&) constexpr {
                return Matrix<decltype(T()*T2()), M, O>{ m1.getRow(IDX / O).dot(m2.getCol(IDX % O))... };
            }(self, m, meta::sequenceList<M*O>());
        }
    };

    // Right-side operator overloads
    template <class STORAGETYPE, typename T, std::size_t... DIMS>
    constexpr auto operator*(const nonArray auto& s, const TensorType<STORAGETYPE, T, DIMS...> &t) { return t.map([&s](const T& e) { return s * e; }); }
    template <class STORAGETYPE, typename T, std::size_t... DIMS>
    constexpr auto operator/(const nonArray auto& s, const TensorType<STORAGETYPE, T, DIMS...> &t) { return t.map([&s](const T& e) { return s / e; }); }
    template <class STORAGETYPE, typename T, std::size_t FIRSTDIM, std::size_t... RESTDIMS>
    constexpr std::ostream& operator<<(std::ostream& os, const TensorType<STORAGETYPE, T, FIRSTDIM, RESTDIMS...>& t) {
        t.template prettyPrint<(RESTDIMS * ... * 1uz), FIRSTDIM, RESTDIMS...>(os, meta::sequenceList<FIRSTDIM>());
        return os;
    }
}
