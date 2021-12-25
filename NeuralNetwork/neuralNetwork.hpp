#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "matrix.hpp"
#include <vector>
#include <math.h>
#include <random>
#include <chrono>
#include <fstream>

template <int inputNodeLength, int hiddenNodeLength, int outputNodeLength>
class NeuralNetwork {
public:
    NeuralNetwork(const double& learningRate)
    :   m_learningRate(learningRate) {

        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::normal_distribution<double> inputHiddenDistribution(0.0, std::pow(hiddenNodeLength, -0.5));
        std::normal_distribution<double> hiddenOutputDistribution(0.0, std::pow(outputNodeLength, -0.5));

        for(int i=0; i<m_weight_IH.getRows(); ++i) {
            for(int j=0; j<m_weight_IH.getCols(); ++j) {
                m_weight_IH[i][j] = inputHiddenDistribution(generator);
            }
        }

        for(int j=0; j<m_weight_HO.getRows(); ++j) {
            for(int k=0; k<m_weight_HO.getCols(); ++k) {
                m_weight_HO[j][k] = hiddenOutputDistribution(generator);
            }
        }
    }

    void train(const Matrix<double, inputNodeLength, 1>& inputs, const Matrix<double, outputNodeLength, 1>& targets) {
        auto hidden_inputs = m_weight_IH.dot(inputs);
        auto hidden_outputs = activate(hidden_inputs);

        auto final_inputs = m_weight_HO.dot(hidden_outputs);
        auto final_outputs = activate(final_inputs);

        m_error = targets - final_outputs;
        auto hidden_errors = m_weight_HO.transpose().dot(m_error);

        m_weight_HO += m_learningRate * (m_error * final_outputs * (1.0 - final_outputs)).dot(hidden_outputs.transpose());
        m_weight_IH += m_learningRate * (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)).dot(inputs.transpose());
    }

    auto query(const Matrix<double, inputNodeLength, 1>& inputs) {
        auto hidden_inputs = m_weight_IH.dot(inputs);
        auto hidden_outputs = activate(hidden_inputs);

        auto final_inputs = m_weight_HO.dot(hidden_outputs);
        auto final_outputs = activate(final_inputs);

        return final_outputs;
    }

    void saveModel(const std::string& fileName) const {
        std::ofstream file(fileName, std::ios::binary);
        file.write((char*)this, sizeof(*this));
    }

    void loadModel(const std::string& fileName) const {
        std::ifstream file(fileName, std::ios::binary);
        file.read((char*)this, sizeof(*this));
    }

    double getError() const {
        double error = 0;
        for(int i=0; i<m_error.getRows(); ++i) {
            error += m_error[i][0] * m_error[i][0];
        }
        return sqrt(error/m_error.getRows());
    }
private:
    double activate(const double& val) {
        return 1/(1 + std::exp(-val));
    }

    template<typename T, int rows, int cols>
    auto activate(const Matrix<T,rows,cols>& matrix) {
        auto mat = matrix;
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<cols; ++j) {
                mat[i][j] = activate(mat[i][j]);
            }
        }
        return mat;
    }
private:
    const double m_learningRate;
    Matrix<double, outputNodeLength, 1> m_error;

    Matrix<double, hiddenNodeLength, inputNodeLength> m_weight_IH;
    Matrix<double, outputNodeLength, hiddenNodeLength> m_weight_HO;
};

#endif