import std;

import linalg;

#define STR_EVAL(...) #__VA_ARGS__ " => " << __VA_ARGS__
#define LINE_EVAL(...) #__VA_ARGS__ " =\n" << __VA_ARGS__


namespace {
    [[maybe_unused]] void testVectors() {
        linalg::Vector v1{{ 1.0, 2.0, 3.0 }};
        std::cout << STR_EVAL(-v1) << "\t<- a 'value-type' vector that owns its data." << std::endl;
        float reallyLongArray[] = { -1.0f, -3.0f, 0.0f, 1.0f, 4.2f, 3.9f, -33.0f, 0.003f, 14.0f, 0.0f, 0.0f, 22.0f };
        linalg::VectorPtr<float, 3uz, -3z> v2{ reallyLongArray + 8uz };
        //                          type   size stride  parent        offset

        // Pointer types
        std::cout << STR_EVAL(v2) << "\t<- a 'pointer-type' vector that doesn't own.\nreallyLongArray=";
        for (const auto &e : reallyLongArray)
            std::cout << " " << e;
        std::cout << std::endl;

        for (auto &e : v2)
            ++e;

        std::cout << "incremented v2's elements.\nreallyLongArray=";
        for (const auto &e : reallyLongArray)
            std::cout << " " << e;
        std::cout << std::endl;

        // Utilities showcase
        v2 -= v1;
        std::cout << "v2-=v1; " STR_EVAL(v2) << "\t" << STR_EVAL(v2*4u) << "\t" << STR_EVAL(v1 + v2) << "\t" STR_EVAL(v1.cross(v2)) << "\t" STR_EVAL(v1.direction()) << std::endl;

        std::cout << STR_EVAL(v1.map([](double a){ return a > 0.0; }).reduce([](bool a, bool b){ return a && b; })) << std::endl;
    }

    [[maybe_unused]] void testMatrices() {
        [[maybe_unused]]
        constexpr linalg::Matrix<double, 2uz, 5uz> m1{ 1.0, 0.0, 1.0, 0.0, 1.0,
                                                       0.0, 1.0, 0.0, 1.0, 0.0 };
        [[maybe_unused]]
        constexpr linalg::Matrix<float, 5uz, 3uz> m2{ 1.0f, 0.0f, 1.0f,
                                                      0.0f, 1.0f, 0.0f,
                                                      1.0f, 0.0f, 1.0f,
                                                      0.0f, 1.0f, 0.0f,
                                                      1.0f, 0.0f, 1.0f };

        // std::cout << m1 << "\n" << std::endl;
        // std::cout << m2 << "\n" << std::endl;
        std::cout << LINE_EVAL(m1 * m2) << std::endl;

        static float reallyLongArray[] = { -1.0f, -3.0f, 0.0f, 1.0f, 4.2f, 3.9f, -33.0f, 0.003f, 14.0f, 0.0f, 0.0f, 22.0f };
        [[maybe_unused]]
        constexpr linalg::MatrixPtr<float, 2uz, 3uz, -2> m3{ reallyLongArray + 10uz };
        std::cout << LINE_EVAL(m3) << std::endl;

        std::cout << STR_EVAL(m3['*', 2][1]) << std::endl;

        auto m4 = linalg::Matrix<std::uint32_t, 5uz, 5uz>::Identity();
        m4.getRow(3uz) += linalg::Vector<std::uint32_t, 5uz>::broadcast(4u);
        m4.getRow(0uz) = m4.getCol(4uz);
        m4.getDiagonal() *= 3u;
        std::cout << LINE_EVAL(m4) << std::endl;
    }

    [[maybe_unused]] void testHigherDims() {
        // Massive 6-dimensional matrix, uses CTAD to deduce template of <ValueType<...>, int, 2, 3, 2, 3, 2, 3>
        [[maybe_unused]]
        linalg::Multidimensional h1{
        { { { { { {  0,  1,  2},
                  {  3,  4,  5} },
                    { {  6,  7,  8},
                      {  9, 10, 11} },
                        { { 12, 13, 14},
                          { 15, 16, 17} } },
              { { { 18, 19, 20},
                  { 21, 22, 23} },
                    { { 24, 25, 26},
                      { 27, 28, 29} },
                        { { 30, 31, 32},
                          { 33, 34, 35} } } },

            { { { { 36, 37, 38},
                  { 39, 40, 41} },
                    { { 42, 43, 44},
                      { 45, 46, 47} },
                        { { 48, 49, 50},
                          { 51, 52, 53} } },
            { { { 54, 55, 56},
                { 57, 58, 59} },
                    { { 60, 61, 62},
                      { 63, 64, 65} },
                        { { 66, 67, 68},
                          { 69, 70, 71} } } },

            { { { { 72, 73, 74},
                  { 75, 76, 77} },
                    { { 78, 79, 80},
                      { 81, 82, 83} },
                        { { 84, 85, 86},
                          { 87, 88, 89} } },
            { { { 90, 91, 92},
                { 93, 94, 95} },
                    { { 96, 97, 98},
                      { 99,100,101} },
                        { {102,103,104},
                          {105,106,107} } } } },


                { { { { {108,109,110},
                        {111,112,113} },
                          { {114,115,116},
                            {117,118,119} },
                              { {120,121,122},
                                {123,124,125} } },
                    { { {126,127,128},
                        {129,130,131} },
                          { {132,133,134},
                            {135,136,137} },
                              { {138,139,140},
                                {141,142,143} } } },

                  { { { {144,145,146},
                        {147,148,149} },
                          { {150,151,152},
                            {153,154,155} },
                              { {156,157,158},
                                {159,160,161} } },
                    { { {162,163,164},
                        {165,166,167} },
                          { {168,169,170},
                            {171,172,173} },
                              { {174,175,176},
                                {177,178,179} } } },

                  { { { {180,181,182},
                        {183,184,185} },
                          { {186,187,188},
                            {189,190,191} },
                              { {192,193,194},
                                {195,196,197} } },
                    { { {198,199,200},
                        {201,202,203} },
                          { {204,205,206},
                            {207,208,209} },
                              { {210,211,212},
                                {213,214,215} } } } } }
        };

        std::cout << STR_EVAL(sizeof(h1)) << " bytes\n" STR_EVAL(h1.count()) << "\n" LINE_EVAL(h1) << "\n" << std::endl;

        std::cout << LINE_EVAL(h1[0, '*', 1, 0]) << std::endl;
        std::cout << "Incrementing all values in slice 'h1[0, '*', 1, 0]'" << std::endl;
        for (auto &elem : h1[0, '*', 1, 0])
            ++elem;
        std::cout << LINE_EVAL(h1[0, '*', 1, 0]) << "\n" << std::endl;

        std::cout << STR_EVAL(++h1[0, 2, 1, 0, 0, 1]) << std::endl;
        std::cout << STR_EVAL(++h1[0][2, 1, 0][0, 1]) << std::endl;
        std::cout << STR_EVAL(++h1[0][2][1][0][0][1]) << std::endl;
        std::cout << STR_EVAL(++h1[0, '*', '*', 0, 0][2, 1, 1]) << std::endl;
        std::cout << STR_EVAL(++h1['*', '*', '*', '*', '*', 1]['*', '*', '*', '*', 0]['*', '*', '*', 0]['*', '*', 1]['*', 2][0]) << std::endl;

        [[maybe_unused]]
        constexpr linalg::Multidimensional<double, 2uz, 2uz, 3uz, 4uz> h2 {
            0,  1,  2,  3,
            4,  5,  6,  7,
            8,  9, 10, 11,
                12, 13, 14, 15,
                16, 17, 18, 19,
                20, 21, 22, 23,
            24, 25, 26, 27,
            28, 29, 30, 31,
            32, 33, 34, 35,
                36, 37, 38, 39,
                40, 41, 42, 43,
                44, 45, 46, 47,
        };

        [[maybe_unused]]
        constexpr auto h3 = linalg::Multidimensional<int, 4uz, 3uz, 2uz, 5uz>::broadcast(1);

        //std::cout << LINE_EVAL(h2.contract<3uz>(h3)) << "\n" LINE_EVAL(h2.contract<2uz>(h3)) << std::endl;

        // Test for compile-time evaluation
        static_assert(h2['*', 1][0, '*', 2][2] > 0.0);
    }
}

//////////
// MAIN //
//////////
int main() {
    // ::testVectors();
    // ::testMatrices();
    ::testHigherDims();

    return 0;
}
