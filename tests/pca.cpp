#include <iostream>
#include <sstream>
#include <vector>

#include "Catch/single_include/catch2/catch.hpp"

#include "header-only-pca/read_data.hpp"
#include "header-only-pca/types.hpp"
#include "header-only-pca/pca.hpp"

TEST_CASE("Test pca::get_covariance") {
    SECTION("...with the 3-identity") {
        pca::Matrix id = matrix_identity(3);

        pca::Matrix cov = pca::get_covariance(id);

        std::vector<double> v{
            1./3, 0, 0,
            0, 1./3, 0,
            0, 0, 1./3
        };

        pca::Matrix res = matrix_from_array(v.data(), 3, 3);

        REQUIRE(matrix_equal(res, cov, 0.001));
    }
    SECTION("...with the 100-identity") {
        pca::Matrix id = matrix_identity(100);

        pca::Matrix cov = pca::get_covariance(id);

        std::vector<double> v(100 * 100, 0);

        for (size_t i = 0; i < 100; ++i) {
            v[100*i + i] = 1./100;
        }

        pca::Matrix res = matrix_from_array(v.data(), 100, 100);

        REQUIRE(matrix_equal(res, cov, 0.0001));
    }
}

TEST_CASE("Test pca::get_mean ") {
    SECTION("...with a trivial case") {
        std::vector<double> v(1, 1);
        pca::Vector a = vector_from_array(v.data(), 1);

        REQUIRE(pca::get_mean (a) == 1.);
    }
    SECTION("...with a simple case") {
        std::vector<double> v(5);
        std::iota(
            v.begin(),
            v.end(),
            1
        );
        pca::Vector a = vector_from_array(v.data(), 5);

        // the mean of {1, 2, 3, 4, 5} is clearly 3
        REQUIRE(pca::get_mean (a) == 3.);
    }
    SECTION("...with a simple, yet long, case") {
        std::vector<double> v(1000);
        std::iota(
            v.begin(),
            v.end(),
            1
        );
        pca::Vector a = vector_from_array(v.data(), 1000);

        // the sum of {1, 2, ..., 1000} is 500500
        // so the mean is 500500 / 1000 = 500.5
        REQUIRE(pca::get_mean (a) == 500.5);
    }
    SECTION("...with a small pca::Vector of integer values") {
        std::vector<double> v{1, 2, 6, 4, 7, 32, 345, 67, 234, 234};
        pca::Vector a = vector_from_array(v.data(), v.size());

        REQUIRE(pca::get_mean (a) == 93.2);
    }
    SECTION("...with a small pca::Vector of decimals") {
        std::vector<double> v{1.3, 23.1, -2.6, 3.235, 7.54, 314.2, 0.33332, 1.32, 2};
        pca::Vector a = vector_from_array(v.data(), v.size());

        using namespace Catch::literals;
        REQUIRE(pca::get_mean (a) == 38.9365_a);
    }
}

TEST_CASE("Test PCA") {
    SECTION("...with a 3x3 matrix") {
    std::vector<double> v{
        5, 4, 1,
        7, 4, 0,
        6, 5, 9
    };

    pca::Matrix v_m = matrix_from_array(v.data(), 3, 3);

    pca::PCA pca(3);

    pca::Matrix res = pca.doPCA(v_m);

    // from Python's sklearn
    std::vector<double> expected{
        -8.19288030e-01, 1.29010990e+00, 9.01824638e-17,
        -1.17041147e+00, -1.14676436e+00, 9.01824638e-17,
        1.98969950e+00, -1.43345545e-01, 9.01824638e-17
    };

    std::vector<double> result(
        DATA(res),
        DATA(res) + res->n_col * res->n_row
    );

    REQUIRE_THAT(result, Catch::Approx(expected).margin(0.000001));
    }

    SECTION("...with a 4x4 matrix") {
    std::vector<double> v{
        3, 2, 4, 1,
        6, 4, 2, 4,
        6, 2, 9, 7,
        2, 6, 9, 2
    };

    pca::Matrix v_m = matrix_from_array(v.data(), 4, 4);

    pca::PCA pca(3);

    pca::Matrix res = pca.doPCA(v_m);

    // from Python's sklearn
    std::vector<double> expected{
        0.64821642, -1.20961317, 1.01811501,
        -0.72044211, -1.02215418, -1.1043944,
        -1.90252228, 1.1243709, 0.41936492,
        1.97474797, 1.10739645, -0.33308553
    };

    std::vector<double> result(
        DATA(res),
        DATA(res) + res->n_col * res->n_row
    );

    REQUIRE_THAT(result, Catch::Approx(expected).margin(0.0001));

    }
}
