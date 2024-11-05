#include "linearAlgebra.h"

using namespace std;

#define STR_EVAL(...) #__VA_ARGS__ "=> " << __VA_ARGS__
#define LINE_EVAL(...) #__VA_ARGS__ "=\n" << __VA_ARGS__


void testVectors() {
    auto v1 = linalg::Vector{{ 1.0, 2.0, 3.0 }};
    cout << STR_EVAL(-v1) << "\t<- a 'value-type' vector that owns its data." << endl;
    float reallyLongArray[] = { -1.0f, -3.0f, 0.0f, 1.0f, 4.2f, 3.9f, -33.0f, 0.003f, 14.0f, 0.0f, 0.0f, 22.0f };
    auto v2 = linalg::VectorPtr<float, 3uz, -3z>{reallyLongArray + 8uz};
    //                          type   size stride  parent        offset

    // Pointer types
    cout << STR_EVAL(v2) << "\t<- a 'pointer-type' vector that doesn't own.\nreallyLongArray=";
    for (const auto &e : reallyLongArray)
        cout << " " << e;
    cout << endl;

    for (auto &e : v2)
        ++e;

    cout << "incremented v2's elements.\nreallyLongArray=";
    for (const auto &e : reallyLongArray)
        cout << " " << e;
    cout << endl;

    // Utilities showcase
    v2 -= v1;
    cout << "v2-=v1; " STR_EVAL(v2) << "\t" << STR_EVAL(v2*4u) << "\t" << STR_EVAL(v1 + v2) << "\t" STR_EVAL(v1.cross(v2)) << "\t" STR_EVAL(v1.direction()) << endl;

    cout << STR_EVAL(v1.map([](double a){ return a > 0.0; }).reduce([](bool a, bool b){ return a && b; })) << endl;
}

void testMatrices() {
    // constexpr auto m1 = linalg::Matrix<double, 2uz, 5uz>{ { { 1.0, 0.0, 1.0, 0.0, 1.0 },
    //                                                         { 0.0, 1.0, 0.0, 1.0, 0.0 } } };
    // cout << m1 << "\n" << endl;
    // constexpr auto m2 = linalg::Matrix<float, 5uz, 3uz>{ { { 1, 0, 1 },
    //                                                        { 0, 1, 0 },
    //                                                        { 1, 0, 1 },
    //                                                        { 0, 1, 0 },
    //                                                        { 1, 0, 1 } } };
    constexpr auto m2 = linalg::Matrix<float, 5uz, 3uz>{ { { 0, 1, 2 },
                                                           { 3, 4, 5 },
                                                           { 6, 7, 8 },
                                                           { 9,10,11 },
                                                           {12,13,14 } } };
    cout << m2 << "\n" << endl;

    // m2[3uz, 2uz] = 0.0f;
    // cout << m2 << "\n" << endl;

    // constexpr auto m3 = m1 * m2;
    // cout << "constexpr " LINE_EVAL(m3) << endl;

    // // Vector x Matrix multiply
    // constexpr linalg::Vector v1{linalg::TensorBase<linalg::ValueType, 1z, float, 4uz>{ 2.0f, 1.0f, 0.0f, -1.0f }};
    // cout << STR_EVAL(v1) << endl;

    // auto m5 = linalg::Matrix<unsigned, 5uz, 5uz>::Identity();
    // m5.getRow(3uz) += linalg::Vector<uint32_t, 5uz>{ 4u };
    // m5.getRow(0uz) = m5.getCol(4uz);
    // m5.getDiagonal() *= 3u;
    // cout << LINE_EVAL(m5) << endl;
}

void testTensors() {
    // Massive 6-dimensional multilinear tensor
    [[maybe_unused]] constexpr auto tensor1 = linalg::Tensor{
        {{{{{{  0,  1,  2},
             {  3,  4,  5}},
                {{  6,  7,  8},
                 {  9, 10, 11}},
                    {{ 12, 13, 14},
                     { 15, 16, 17}}},
           {{{ 18, 19, 20},
             { 21, 22, 23}},
                {{ 24, 25, 26},
                 { 27, 28, 29}},
                    {{ 30, 31, 32},
                     { 33, 34, 35}}}},

          {{{{ 36, 37, 38},
             { 39, 40, 41}},
                {{ 42, 43, 44},
                 { 45, 46, 47}},
                    {{ 48, 49, 50},
                     { 51, 52, 53}}},
           {{{ 54, 55, 56},
             { 57, 58, 59}},
                {{ 60, 61, 62},
                 { 63, 64, 65}},
                    {{ 66, 67, 68},
                     { 69, 70, 71}}}},

          {{{{ 72, 73, 74},
             { 75, 76, 77}},
                {{ 78, 79, 80},
                 { 81, 82, 83}},
                    {{ 84, 85, 86},
                     { 87, 88, 89}}},
           {{{ 90, 91, 92},
             { 93, 94, 95}},
                {{ 96, 97, 98},
                 { 99,100,101}},
                    {{102,103,104},
                     {105,106,107}}}}},


             {{{{{108,109,110},
                 {111,112,113}},
                    {{114,115,116},
                     {117,118,119}},
                        {{120,121,122},
                         {123,124,125}}},
               {{{126,127,128},
                 {129,130,131}},
                    {{132,133,134},
                     {135,136,137}},
                        {{138,139,140},
                         {141,142,143}}}},

              {{{{144,145,146},
                 {147,148,149}},
                    {{150,151,152},
                     {153,154,155} },
                        {{156,157,158},
                         {159,160,161}}},
               {{{162,163,164},
                 {165,166,167}},
                    {{168,169,170},
                     {171,172,173}},
                        {{174,175,176},
                         {177,178,179}}}},

              {{{{180,181,182},
                 {183,184,185}},
                    {{186,187,188},
                     {189,190,191}},
                        {{192,193,194},
                         {195,196,197}}},
               {{{198,199,200},
                 {201,202,203}},
                    {{204,205,206},
                     {207,208,209}},
                        {{210,211,212},
                         {213,214,215}}}}}}
    };

    // Test for compile-time evaluation
    //static_assert(tensor1[0][2, 1, 0][0, 1] > 0);
    //cout << LINE_EVAL(tensor1) << "\n" STR_EVAL(sizeof(tensor1)) << "\n" STR_EVAL(tensor1[0, 2, 1, 0, 0, 1]) << "\n" STR_EVAL(tensor1[0][2][1][0][0][1]) << endl;
}

//////////
// MAIN //
//////////
int main() {
    // testVectors();
    testMatrices();
    // testTensors();

    return 0;
}
