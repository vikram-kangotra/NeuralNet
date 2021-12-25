#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>

template <typename T, int rows = 3, int cols = 3>
class Matrix
{
public:
    Matrix(const T& all = 0) {
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<cols; ++j) {
                m_data[i][j] = all;
            }
        }
    }

    Matrix(const std::initializer_list<T>& list) {

        if(list.size() != rows*cols) {
            std::cout << "length error";
            exit(0);
        }

        int index = 0;
        for(const auto& elem: list) {
            m_data[index/cols][index%cols] = elem;
            ++index;
        }
    }

    Matrix(const std::vector<T>& list) {

        if(list.size() != rows*cols) {
            std::cout << "length error";
            exit(0);
        }

        for(int i=0; i<list.size(); ++i) {
            m_data[i/cols][i%cols] = list[i];
        }
    }

    static Matrix identity() {
        Matrix<T, rows, cols> iden;
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<cols; ++j) {
                if(i==j) 
                    iden[i][j] = 1;
                else
                    iden[i][j] = 0;
            }
        }
        return iden;
    }

    Matrix<T,cols,rows> transpose() const {
        Matrix<T, cols,rows> matrix;
        for(int i=0; i<cols; ++i) {
            for(int j=0; j<rows; ++j) {
                matrix[i][j] = (*this)[j][i];
            }
        }
        return matrix;
    }

    int getRows() const {
        return rows;
    }

    int getCols() const {
        return cols;
    }

    Matrix<T,rows,cols> operator-() const {
        auto matrix = *this;
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<cols; ++j) {
                matrix[i][j] = -matrix[i][j];
            }
        }
        return matrix;
    }

    Matrix<T,rows,cols> operator+(const T& val) const {
        auto mat = *this;
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<cols; ++j) {
                mat[i][j] += val;
            }
        }
        return mat;
    }

    Matrix<T,rows,cols> operator-(const T& val) const {
        auto mat = *this;
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<cols; ++j) {
                mat[i][j] -= val;
            }
        }
        return mat;
    }

    Matrix<T,rows,cols> operator*(const T& val) const {
        auto mat = *this;
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<cols; ++j) {
                mat[i][j] *= val;
            }
        }
        return mat;
    }

    Matrix<T,rows,cols>& operator+=(const Matrix<T,rows,cols>& other) {
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<cols; ++j) {
                (*this)[i][j] += other[i][j];
            }
        }
        return *this;
    }

    Matrix<T,rows,cols> operator+(const Matrix<T,rows,cols>& other) const {
        auto mat = *this;
        return mat += other;
    }

    Matrix<T,rows,cols>& operator-=(const Matrix<T,rows,cols>& other) {
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<cols; ++j) {
                (*this)[i][j] -= other[i][j];
            }
        }
        return *this;
    }

    Matrix<T,rows,cols> operator-(const Matrix<T,rows,cols>& other) const {
        auto mat = *this;
        return mat -= other;
    }

    Matrix<T,rows,cols> operator*=(const Matrix<T,rows,cols>& other) {
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<cols; ++j) {
                (*this)[i][j] *= other[i][j];
            }
        }
        return *this;
    }

    Matrix<T,rows,cols> operator*(const Matrix<T,rows,cols>& other) const {
        auto mat = *this;
        return mat *= other;
    }

    template<int otherCols>
    Matrix<T,rows,otherCols> dot(const Matrix<T,cols,otherCols>& other) const {
        Matrix<T,rows,otherCols> mat(0);
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<otherCols; ++j) {
                for(int k=0; k<cols; ++k) {
                    mat[i][j] += (*this)[i][k] * other[k][j];
                }
            }
        }
        return mat;
    }

private:
    class HelperIndexer {
    public:
        HelperIndexer(const T* const_m_data)
        : const_m_data(const_m_data) {}

        HelperIndexer(T* m_data)
        : m_data(m_data) {}

        const T& operator[](const int& j) const {
            return const_m_data[j];
        }

        T& operator[](const int& j) {
            return m_data[j];
        }
    private:
        T* m_data;
        const T* const_m_data;
    };

public:
    const HelperIndexer operator[](const int& i) const {
        return HelperIndexer(m_data[i]);
    }

    HelperIndexer operator[](const int& i) {
        return HelperIndexer(m_data[i]);
    }

private:
    T m_data[rows][cols];
};

template<typename T, int rows, int cols>
std::ostream& operator<<(std::ostream& os, const Matrix<T,rows,cols>& matrix)
{
    for(int i=0; i<rows; ++i) {
        os << "| ";
        for(int j=0; j<cols; ++j) {
            os << matrix[i][j] << ' ';
        }
        os << "|\n";
    }
    return os;
}

template<typename T, int rows, int cols>
Matrix<T,rows,cols> operator*(const T& val, const Matrix<T,rows,cols>& other) {
    auto matrix = other;
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            matrix[i][j] *= val;
        }
    }
    return matrix;
}

template<typename T, int rows, int cols>
Matrix<T,rows,cols> operator+(const T& val, const Matrix<T,rows,cols>& other) {
    auto matrix = other;
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            matrix[i][j] += val;
        }
    }
    return matrix;
}

template<typename T, int rows, int cols>
Matrix<T,rows,cols> operator-(const T& val, const Matrix<T,rows,cols>& other) {
    auto matrix = other;
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            matrix[i][j] = val - matrix[i][j];
        }
    }
    return matrix;
}

#endif