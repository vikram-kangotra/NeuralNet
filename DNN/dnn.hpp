#ifndef DNN_HPP
#define DNN_HPP

#include "matrix.hpp"
#include "dnnModel.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <exception>

class DNN {
private:
    DNN() {}
public:

    DNN(const std::vector<int>& topology, const double& learningRate = 0.1)
    :   m_learningRate(learningRate),
        m_weights(topology.size()-1),
        m_outputs(m_weights.size()) {

        if(topology.size() < 2) {
            throw std::length_error("Network needs atleast two layers.");
        }

        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);

        for(int i=0; i<m_weights.size(); ++i) {
            std::normal_distribution<double> distribution(0.0, std::pow(topology[i+1], -0.5));
            
            m_weights[i] = Matrix<double>(topology[i+1],topology[i]);

            for(int j=0; j<topology[i+1]; ++j) {
                for(int k=0; k<topology[i]; ++k) {
                    m_weights[i][j][k] = distribution(generator);
                }
            }
        }
    }

    DNN(const DnnModel& model)
    :   m_learningRate(model.m_learningRate),
        m_weights(model.m_weights),
        m_outputs(model.m_weights.size()) {}

    void setLearningRate(const double& lr) {
        m_learningRate = lr;
    }

    double getError() const {
        double error = 0;
        for(int i=0; i<m_error.getRows(); ++i) {
            error += m_error[i][0] * m_error[i][0];
        }
        return sqrt(error/m_error.getRows());
    }

    void train(const Vertex<double>& input_list, const Vertex<double>& target_list) {
        query(input_list);
        backpropogate(input_list, target_list);
    }

    const Matrix<double>& query(const Vertex<double>& input_list) {
        auto input = m_weights[0].dot(input_list);
        m_outputs[0] = activate(input);

        for(int i=1; i<m_weights.size(); ++i) {
            input = m_weights[i].dot(m_outputs[i-1]);
            m_outputs[i] = activate(input); 
        }

        return m_outputs.back();
    }

    Matrix<double> reverse_query(const Vertex<double>& input_list) {

        auto input = m_weights.back().transpose().dot(input_list);
        auto output = reverseActivate(input);

        for(int i=m_weights.size()-2; i>=0; --i) {
            input = m_weights[i].transpose().dot(output);
            output = reverseActivate(input); 
        }

        return output;
    }

    void saveModel(const std::string& fileName) const {
        DnnModel model{m_learningRate, m_weights};
        model.saveModel(fileName);
    }
private:
    double expit(const double& val) {
        return 1/(1 + std::exp(-val));
    }

    double logit(const double& val) {
        auto value = (val<=0)? 0.01: (val>=1)? 0.99: val;
        return  std::log(std::abs(value/(1-value)));
    }

    template<typename T>
    Matrix<T> activate(const Matrix<T>& matrix) {
        auto mat = matrix;
        for(int i=0; i<mat.getRows(); ++i) {
            for(int j=0; j<mat.getCols(); ++j) {
                mat[i][j] = expit(mat[i][j]);
            }
        }
        return mat;
    }

    template<typename T>
    Matrix<T> reverseActivate(const Matrix<T>& matrix) {
        auto mat = matrix;
        for(int i=0; i<mat.getRows(); ++i) {
            for(int j=0; j<mat.getCols(); ++j) {
                mat[i][j] = logit(mat[i][j]);
            }
        }
        return mat;
    }

    void backpropogate(const Vertex<double>& input_list, const Vertex<double>& target_list) {
        auto error = target_list - m_outputs.back();
        m_error = error;

        for(int i=m_weights.size()-1; i>=1; --i) {
            m_weights[i] += m_learningRate * (error * m_outputs[i] * (1.0 - m_outputs[i])).dot(m_outputs[i-1].transpose());
            error = m_weights[i].transpose().dot(error);
        }

        m_weights[0] += m_learningRate * (error * m_outputs[0] * (1.0 - m_outputs[0])).dot(input_list.transpose());
    }
private:
    double m_learningRate;
    Matrix<double> m_error;
    std::vector<Matrix<double>> m_weights;
    std::vector<Matrix<double>> m_outputs;
};

#endif
