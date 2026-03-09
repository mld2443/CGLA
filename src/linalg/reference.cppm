export module linalg:reference;

import std;

import meta;

export namespace linalg {
    template <class STORAGEBASE, typename T, std::size_t... DIMS>
    struct TensorType;

    // Reference class, transient and transparent
    template <class PARENTCLASS, std::size_t C, std::size_t... DIMSANDSTEPS>
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

        template <class SELF>
        constexpr decltype(auto) deref(this SELF&& self, auto first, auto... inds) {
            return []<std::size_t THISDIM, std::size_t THISSTEP, std::size_t... RESTDIMSSTEPS, std::size_t... NEWDIMSANDSTEPS, std::size_t... NEWDIMS>(
                        this auto getSubstruct, SELF&& self,
                        meta::List<NEWDIMSANDSTEPS...>&&, meta::List<NEWDIMS...>&&,
                        std::size_t offset, auto nextInd, auto... restInds) constexpr -> decltype(auto) {
                if constexpr (std::is_same_v<decltype(nextInd), char>) { // nextInd is a wildcard
                    if constexpr (sizeof...(restInds))             // more given indices after this wildcard
                        return getSubstruct.template operator()<RESTDIMSSTEPS...>(std::forward<SELF>(self), meta::List<NEWDIMSANDSTEPS..., THISDIM, THISSTEP>{}, meta::List<NEWDIMS..., THISDIM>{}, offset, restInds...);
                    else if constexpr (sizeof...(RESTDIMSSTEPS))   // remaining dimensions are implied wildcards
                        return getSubstruct.template operator()<RESTDIMSSTEPS...>(std::forward<SELF>(self), meta::List<NEWDIMSANDSTEPS..., THISDIM, THISSTEP>{}, meta::List<NEWDIMS..., THISDIM>{}, offset, '*');
                    else                                           // final index was given or implied wildcard
                        return TensorType<ReferenceClass<PARENTCLASS, (NEWDIMS * ... * THISDIM), NEWDIMSANDSTEPS..., THISDIM, 1uz>, meta::copyConst<SELF, std::remove_cvref_t<decltype(*self.ref.data)>>, NEWDIMS..., THISDIM>(self.ref, offset);
                } else {
                    offset += THISSTEP * static_cast<std::size_t>(nextInd);
                    if constexpr (sizeof...(restInds))             // more constraints to get through and/or there are unconstrained dimensions
                        return getSubstruct.template operator()<RESTDIMSSTEPS...>(std::forward<SELF>(self), meta::List<NEWDIMSANDSTEPS...>{}, meta::List<NEWDIMS...>{}, offset, restInds...);
                    else if constexpr (sizeof...(RESTDIMSSTEPS))   // remaining dimensions are implied wildcards
                        return getSubstruct.template operator()<RESTDIMSSTEPS...>(std::forward<SELF>(self), meta::List<NEWDIMSANDSTEPS...>{}, meta::List<NEWDIMS...>{}, offset, '*');
                    else if constexpr (sizeof...(NEWDIMSANDSTEPS)) // all indices were given but at least one was a wildcard
                        return TensorType<ReferenceClass<PARENTCLASS, (NEWDIMS * ...), NEWDIMSANDSTEPS...>, meta::copyConst<SELF, std::remove_cvref_t<decltype(*self.ref.data)>>, NEWDIMS...>(self.ref, offset);
                    else                                           // all indices given, no wildcards
                        return self.ref.get(offset);
                }
            }.template operator()<DIMSANDSTEPS...>(std::forward<SELF>(self), {}, {}, self.offset, first, inds...);
        }

    public:
        constexpr ReferenceClass(PARENTCLASS& ref, std::size_t offset) : ref(ref), offset(offset) {}

    protected:
        PARENTCLASS& ref;
        std::size_t offset;
    };
}
