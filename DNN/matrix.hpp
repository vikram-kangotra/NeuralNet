#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <memory>
#include <vector>

template<typename T>
class Matrix {
public:
    Matrix(const int& rows = 1, const int& cols = 1) 
    :   m_rows(rows),
        m_cols(cols),
        m_data(std::make_unique<T[]>(m_rows*m_cols)) {}

    Matrix(const Matrix<T>& other) {
       *this = other;
    }

    Matrix<T>& operator=(const Matrix<T>& other) {

        m_rows = other.getRows();
        m_cols = other.getCols();
        m_data = std::make_unique<T[]>(m_rows * m_cols);

        for(int i=0; i<m_rows; ++i) {
            for(int j=0; j<m_cols; ++j) {
                (*this)[i][j] = other[i][j];
            }
        }

        return *this;
    }

    template<std::size_t N>
    void set(const T(&list)[N]) {
        if(N != m_rows*m_cols) {
            throw std::length_error("more elements were provided than the holding capacity of the matrix");
        }

        for(int i=0; i<N; ++i) {
           m_data[i] = list[i];
        }
    }

    const int& getRows() const {
        return m_rows;
    }

    const int& getCols() const {
        return m_cols;
    }

    static auto identity(const int& rows, const int& cols) {
        Matrix<T> iden(rows, cols);
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<cols; ++j) {
		iden[i][j] = (i==j);
            }
        }
        return iden;
    }

    auto transpose() const {
        Matrix<T> matrix(m_cols, m_rows);
        for(int i=0; i<m_cols; ++i) {
            for(int j=0; j<m_rows; ++j) {
                matrix[i][j] = (*this)[j][i];
            }
        }
        return matrix;
    }

    auto operator-() const {
        auto matrix = *this;
        for(int i=0; i<m_rows; ++i) {
            for(int j=0; j<m_cols; ++j) {
                matrix[i][j] = -matrix[i][j];
            }
        }
        return matrix;
    }

    auto operator+(const T& val) const {
        auto mat = *this;
        for(int i=0; i<m_rows; ++i) {
            for(int j=0; j<m_cols; ++j) {
                mat[i][j] += val;
            }
        }
        return mat;
    }

    auto operator-(const T& val) const {
        auto mat = *this;
        for(int i=0; i<m_rows; ++i) {
            for(int j=0; j<m_cols; ++j) {
                mat[i][j] -= val;
            }
        }
        return mat;
    }

    auto operator*(const T& val) const {
        auto mat = *this;
        for(int i=0; i<m_rows; ++i) {
            for(int j=0; j<m_cols; ++j) {
                mat[i][j] *= val;
            }
        }
        return mat;
    }

    auto& operator+=(const Matrix<T>& other) {
        for(int i=0; i<m_rows; ++i) {
            for(int j=0; j<m_cols; ++j) {
                (*this)[i][j] += other[i][j];
            }
        }
        return *this;
    }

    auto operator+(const Matrix<T>& other) const {
        auto mat = *this;
        return mat += other;
    }

    auto& operator-=(const Matrix<T>& other) {
        for(int i=0; i<m_rows; ++i) {
            for(int j=0; j<m_cols; ++j) {
                (*this)[i][j] -= other[i][j];
            }
        }
        return *this;
    }

    auto operator-(const Matrix<T>& other) const {
        auto mat = *this;
        return mat -= other;
    }

    auto operator*=(const Matrix<T>& other) {
        for(int i=0; i<m_rows; ++i) {
            for(int j=0; j<m_cols; ++j) {
                (*this)[i][j] *= other[i][j];
            }
        }
        return *this;
    }

    auto operator*(const Matrix<T>& other) const {
        auto mat = *this;
        return mat *= other;
    }

    auto dot(const Matrix<T>& other) const {
        if(this->m_cols != other.getRows()) {
            throw std::length_error("mismatched matrix for dot product. Dimensions not correct\n");
        }

        Matrix<T> mat(m_rows, other.getCols());
        for(int i=0; i<m_rows; ++i) {
            for(int j=0; j<other.getCols(); ++j) {
                for(int k=0; k<m_cols; ++k) {
                    mat[i][j] += (*this)[i][k] * other[k][j];
                }
            }
        }
        return mat;
    }
private:
    class HelperIndexer {
    public:
        HelperIndexer(const T* const_data)
        : m_const_data(const_data) {}

        HelperIndexer(T* data)
        : m_data(data) {}

        const T& operator[](const int& j) const {
            return m_const_data[j];
        }

        T& operator[](const int& j) {
            return m_data[j];
        }
    private:
        T* m_data;
        const T* m_const_data;
    };

public:
    const HelperIndexer operator[](const int& i) const {
        return HelperIndexer((const T*)&m_data[i*m_cols]);
    }

    HelperIndexer operator[](const int& i) {
        return HelperIndexer(&m_data[i*m_cols]);
    }

private:
    int m_rows, m_cols;
    std::unique_ptr<T[]> m_data;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix)
{
    for(int i=0; i<matrix.getRows(); ++i) {
        os << "| ";
        for(int j=0; j<matrix.getCols(); ++j) {
            os << matrix[i][j] << ' ';
        }
        os << "|\n";
    }
    return os;
}

template<typename T>
auto operator*(const T& val, const Matrix<T>& other) {
    auto matrix = other;
    for(int i=0; i<matrix.getRows(); ++i) {
        for(int j=0; j<matrix.getCols(); ++j) {
            matrix[i][j] *= val;
        }
    }
    return matrix;
}

template<typename T>
auto operator+(const T& val, const Matrix<T>& other) {
    auto matrix = other;
    for(int i=0; i<matrix.getRows(); ++i) {
        for(int j=0; j<matrix.getCols(); ++j) {
            matrix[i][j] += val;
        }
    }
    return matrix;
}

template<typename T>
auto operator-(const T& val, const Matrix<T>& other) {
    auto matrix = other;
    for(int i=0; i<matrix.getRows(); ++i) {
        for(int j=0; j<matrix.getCols(); ++j) {
            matrix[i][j] = val - matrix[i][j];
        }
    }
    return matrix;
}

template<typename T>
class Vertex : public Matrix<T> {
public:
    Vertex(const int& rows)
    :   Matrix<T>(rows, 1) {}

    Vertex(const std::initializer_list<T>& list) 
    :   Vertex(list.size()) {
        int i = 0;
        for(const auto& elem: list) {
            (*this)[i][0] = elem;
            ++i;
        }
    }

    Vertex(const std::vector<T>& list) 
    :   Vertex(list.size()) {
        for(int i=0; i<list.size(); ++i) {
            (*this)[i][0] = list[i];
        }
    }
};

#endif