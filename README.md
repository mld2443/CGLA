# CppGLA

A C++ generic linear algebra header library designed for flexibility, readability, and comprehension.

Linear Algebra in C++ is often messy, verbose, cluttered, and/or opaque. Inspired specifically by the source code given for the single value decomposition by [Numerical Recipes in C++](https://numerical.recipes/book.html), I set out to create a tool to help me comprehend what's happening inside before [C++26's \<linalg\> header](https://en.cppreference.com/w/cpp/header/linalg) comes to deliver us from anarchy.

## Design Goals

1. "It just works" - Flexibility and transparency from implementation, usage should look as close to formulae and equations as C++ allows
1. Compile-time evaluation and inlining as much as possible, make your compiler work for you
1. As generic as possible, multilinear tensors of arbitrary dimensionality and size, even if I don't know what to do with them
1. As clean and compact as I can make it

### Things it does not do

1. Replace BLAS - use BLAS if speed is paramount
1. Utilize vector registers or specialty SIMD instructions
1. Sparse data representation, modeling, or optimizations

## Tasks

- [x] SETTLE ON PARADIGM: It's multilinear tensors all the way down and recursive inheritance
- [x] Generic tensor accessor
- [x] Display arbitrary tensors
- [x] Shared ops: map(x), fold(x), scalar mult/div(x), negate(x), add(x), subtract(x), maybe inline mult/div(1)
- [x] Vector ops: dot (x), cross (x)
- [x] Rewrite Matrix and Vector as type aliases; clang 18 doesn't support template deductions for aliases
- [x] Merge StorageBase features into Tensor and consider removing it(x)
- [x] Default initialization(x)
- [x] Rework "internal" functions into lambdas(x)
- [x] Merge recursive ValueType into Tensor(x) and ensure base functionality(x)
- [x] Slicing using the operator[] with a wildcard(x)
- [ ] Change name? Unsure if specifying multilinear tensor is still technically correct
- [ ] Reimpl getRow(1), getCol(1), getDiag(2)
- [ ] Matrix transpose(3)
- [ ] Vector transpose -> Matrix(1)
- [ ] Matrix ops: matrix mult (x), eigenvector/value(3), determinant(3), invert(1), identity(1), rank(3), ...
