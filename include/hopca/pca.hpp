#pragma once

#include <stddef.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <functional>


#include "../../third-party/linalg/single-include/linalg.hpp"


#include "types.hpp"
#include "eigenmap.hpp"

namespace hopca {

inline hopca::Matrix get_covariance(const hopca::Matrix m) {
    hopca::Matrix ret = hola::matrix_multiply_MtN(m, m);

    std::transform(
        DATA(ret),
        DATA(ret) + ret->n_col * ret->n_row,
        DATA(ret),
        [n = ret->n_col](const double a) {
            return a / n;
        }
    );

    return ret;
}

inline double get_mean(const hopca::Vector v) {
    const size_t len = v->length;
    double sum = 0;

    for (size_t i = 0; i < len; ++i) {
        sum += VECTOR_IDX_INTO(v, i);
    }

    return sum / len;
}

class PCA {
    size_t n_components;
    double tolerance;
    size_t max_iter;

public:
    PCA(
        size_t _n_components,
        double _tolerance = 0.0001,
        size_t _max_iter = 1000
    )
    : n_components(_n_components)
    , tolerance(_tolerance)
    , max_iter(_max_iter)
    { ; }

private:
    hopca::Matrix normalize(hopca::Matrix m) const;
public:
    void setNComponents(const size_t n) { n_components = n; }
    void setTolerance(const double d) { tolerance = d; }
    void setMaxIter(const size_t n) { max_iter = n; }

    hopca::Matrix doPCANoNormalize(hopca::Matrix m) const;
    hopca::Matrix doPCA(hopca::Matrix m) const;
};

hopca::Matrix PCA::normalize(hopca::Matrix m) const {
    const size_t rows = m->n_row;
    const size_t cols = m->n_col;

    hopca::Matrix temp = hola::matrix_new(rows, cols);
    assert(temp != nullptr);

    for (size_t col = 0; col < cols; ++col) {
        hopca::Vector vec = hola::matrix_column_copy(m, col);

        const hopca::Vector mean_vector
          = hola::vector_constant(cols, get_mean(vec));
        hola::vector_subtract_into(vec, vec, mean_vector);

        const double norm_squared = hola::vector_dot_product(vec, vec);
        const double invstdev = std::sqrt(rows / norm_squared);

        hola::vector_scalar_multiply_into(vec, vec, invstdev);

        hola::matrix_copy_vector_into_column(temp, vec, col);
    }
    return temp;
}

hopca::Matrix PCA::doPCANoNormalize(hopca::Matrix m) const {

    // find opposite of matrix
    std::transform(
        DATA(m),
        DATA(m) + m->n_col * m->n_row,
        DATA(m),
        [](const double a) {
            return -a;
        }
    );

    const hopca::Matrix cov = get_covariance(m);

    const hopca::Eigen eig = hola::eigen_solve(cov, tolerance, max_iter);

    // map of eigenvalues to eigenvectors
    hopca::Eigenmap eigenmap(eig);
    eigenmap.trim(n_components);

    return hola::matrix_multiply(m, eigenmap.get_matrix());
}

hopca::Matrix PCA::doPCA(hopca::Matrix m) const {
    const hopca::Matrix normalized = normalize(m);
    return doPCANoNormalize(normalized);
}

} // namespace hopca
