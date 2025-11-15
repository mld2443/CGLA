# CppGLA: _An Abuse of Templates_

A C++ generic linear algebra header library designed for flexibility, readability, and comprehension.

Linear Algebra in C/C++ is often messy, verbose, cluttered, and/or opaque. Inspired specifically by the source code given for the single value decomposition by [Numerical Recipes in C++](https://numerical.recipes/book.html), I set out to create a tool to help me comprehend what's happening inside before [C++26's \<linalg\> header](https://en.cppreference.com/w/cpp/header/linalg) comes to deliver us from anarchy.

## Design Goals

1. Simplicity: No need to specify types or dimensions, everything is deduced for you at compile-time
1. Expressiveness: Makes math look like math for improved readability
1. Generic: including 'view' style mutating references to larger data types for maximum utility
1. Compile-time compatible: Any hardcoded values, operations, dereferences, references, etc. will be inlined
1. Multidimensional: Matrices and arrays of arbitrary dimensionality and size, for no reason other than I can!
1. Neatness: as clean and readable as a "template abuse" playgroud can be.

### Examples
```cpp
// start with identity in row-major matrix
auto matrix = linalg::Matrix<unsigned, 5uz, 5uz>::Identity();

// Add a the vector [3 3 3 3 3] to row [3]
matrix[3uz] += linalg::Vector<unsigned, 5uz>::broadcast(3u);

// Set the values of row [0] to the values of column [4]
matrix[0uz] = matrix['*', 4uz];

// Double the values along the diagonal
matrix.getDiagonal() *= 2u;

std::cout << matrix << std::endl;
// 0, 0, 0, 3, 1,
// 0, 2, 0, 0, 0,
// 0, 0, 2, 0, 0,
// 3, 3, 3, 8, 3,
// 0, 0, 0, 0, 2
```
```cpp
// Value initialize and types/dimensions are automatic
auto sixD = linalg::Multidimensional {
  {  {  {  {  {  {
    // ...
    // massive 6-dimensional nested array of data
    // ...
  }, }, }, }, }, },
};

// All of these dereference and increment the exact same index using different slicings
++sixD[0, 2, 1, 0, 0, 1];
++sixD[0][2, 1, 0][0, 1];
++sixD[0][2][1][0][0][1];
++sixD[0, '*', '*', 0, 0][2, 1, 1];
++sixD['*', '*', '*', '*', '*', 1]['*', '*', '*', '*', 0]['*', '*', '*', 0]['*', '*', 1]['*', 2][0];

// Increment all elements in the 3-dimensional subslice matching indices [0, *, 1, 0, *, *]
for (auto &elem : sixD[0, '*', 1, 0])
    ++elem;

// All operations on hard-coded data like the above are compile-time
static_assert(sixD[0, '*', 1, '*', 0][2, 0, 1] > 0);
```

### Things it does not do

1. Replace BLAS - use BLAS if performance is paramount, or the aforementioned C++26 \<linalg\> header once available
1. Utilize vector registers or specialty SIMD instructions
1. Sparse data representation, modeling, or optimizations
1. Capture every elementary operation for a vector/matrix/tensor... yet

## Todo

- Pure, python-style slicing per dimension(4)
- Tensor contraction(4)?
- Matrix transpose(3)
- Covector -> Matrix(1)
- Matrix ops: eigenvector/value(3), minor(3), determinant(1), invert(1), rank(3), ...
- Convert source to unit testing(4)
