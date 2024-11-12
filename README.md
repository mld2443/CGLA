# CppGLA

A C++ generic linear algebra header library designed for flexibility, readability, and comprehension.

Linear Algebra in C++ is often messy, verbose, cluttered, and/or opaque. Inspired specifically by the source code given for the single value decomposition by [Numerical Recipes in C++](https://numerical.recipes/book.html), I set out to create a tool to help me comprehend what's happening inside before [C++26's \<linalg\> header](https://en.cppreference.com/w/cpp/header/linalg) comes to deliver us from anarchy.

## Design Goals

1. "It just works" - Flexibility and transparency from implementation, usage should look as close to formulae and equations as C++ allows
1. Compile-time evaluation and inlining as much as possible, make your compiler work for you
1. As generic as possible, multilinear tensors of arbitrary dimensionality and size, even if I don't know what to do with them
1. As clean and compact as I can make it

### Things it does not do

1. Replace BLAS - use BLAS if speed is paramount, or the aforementioned C++ \<linalg\> header once available
1. Utilize vector registers or specialty SIMD instructions
1. Sparse data representation, modeling, or optimizations
1. Capture every elementary operation for a vector/matrix/tensor, yet
1. Get the names for everything right, probably

## Todo

- Pure, python-style slicing per dimension(4)
- Tensor contraction(4)
- Matrix transpose(3)
- Covector -> Matrix(1)
- Matrix ops: eigenvector/value(3), determinant(3), invert(1), rank(3), ...
