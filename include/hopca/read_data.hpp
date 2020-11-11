#pragma once

#include <iostream>
#include <sstream>

#include <vector>

#include "../../../third-party/linalg/single-include/linalg.hpp"

namespace pca {

inline struct matrix* read_data(const size_t m, const size_t n, std::istream& is = std::cin) {
    double temp;

    std::vector<double> D;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            is >> temp;
            D.push_back(temp);
        }
    }

    struct matrix* res = matrix_from_array(D.data(), m, n);

    return res;
}

} // namespace pca
