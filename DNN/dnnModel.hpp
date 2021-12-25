#ifndef DNN_MODEL_HPP
#define DNN_MODEL_HPP

#include "matrix.hpp"
#include <vector>
#include <fstream>

struct DnnModel {

    void saveModel(const std::string& fileName) const {
        auto index = fileName.find_last_of('.');
        auto extension = fileName.substr(index+1);
        if(extension == "rwm") {
            saveRawModel(fileName);
        } else if(extension == "ftm") {
            saveFormatedModel(fileName);
        } else {
            throw std::invalid_argument(extension + ": no file format for extension");
        }
    }

    static auto loadModel(const std::string& fileName) {
        auto index = fileName.find_last_of('.');
        auto extension = fileName.substr(index+1);
        if(extension == "rwm") {
            return loadRawModel(fileName);
        } else if(extension == "ftm") {
            return loadFormatedModel(fileName);
        } else {
            throw std::invalid_argument(extension + ": no file format for extension");
        }
    }
    
private:
    void saveRawModel(const std::string& fileName) const {
        std::ofstream file(fileName);
        file.exceptions(std::ios::failbit | std::ios::badbit);

        file.write((char*)&m_learningRate, sizeof(m_learningRate));
        int size = m_weights.size();
        file.write((char*)&size, sizeof(size));
        
        for(const auto& weight: m_weights) {
            file.write((char*)&weight.getRows(), sizeof(weight.getRows()));
            file.write((char*)& weight.getCols(), sizeof( weight.getCols()));
            for(int i=0; i<weight.getRows(); ++i) {
                for(int j=0; j<weight.getCols(); ++j) {
                    file.write((char*)&weight[i][j], sizeof(weight[i][j]));
                }
            }
        }
    }

    void saveFormatedModel(const std::string& fileName) const {
        std::ofstream file(fileName);
        file.exceptions(std::ios::failbit | std::ios::badbit);

        file << m_learningRate << " " << m_weights.size() << '\n';
        for(const auto& weight: m_weights) {
            file << weight.getRows() << ' ' << weight.getCols() << '\n';
            for(int i=0; i<weight.getRows(); ++i) {
                for(int j=0; j<weight.getCols(); ++j) {
                    file << weight[i][j] << ' ';
                }
                file << '\n';
            }
        }
    }

    static DnnModel loadRawModel(const std::string& fileName) {
        std::ifstream file(fileName);
        file.exceptions(std::ios::failbit | std::ios::badbit);

        DnnModel model;
        file.read((char*)&model.m_learningRate, sizeof(model.m_learningRate));
        int weightCount;
        file.read((char*)&weightCount, sizeof(weightCount));
        int rows, cols;
        for(int i=0; i<weightCount; ++i) {
            file.read((char*)&rows, sizeof(rows));
            file.read((char*)&cols, sizeof(cols));
            Matrix<double> weight(rows, cols);
            for(int j=0; j<rows; ++j) {
                for(int k=0; k<cols; ++k) {
                    file.read((char*)&weight[j][k], sizeof(weight[j][k]));
                }
            }
            model.m_weights.push_back(weight);
        }
        return model;
    }

    static DnnModel loadFormatedModel(const std::string& fileName) {
        std::ifstream file(fileName);
        file.exceptions(std::ios::failbit | std::ios::badbit);

        DnnModel model;
        file >> model.m_learningRate;
        int weightCount;
        file >> weightCount;
        int rows, cols;
        for(int i=0; i<weightCount; ++i) {
            file >> rows >> cols;
            Matrix<double> weight(rows, cols);
            for(int j=0; j<rows; ++j) {
                for(int k=0; k<cols; ++k) {
                    file >> weight[j][k];
                }
            }
            model.m_weights.push_back(weight);
        }
        return model;
    }
public:
    double m_learningRate;
    std::vector<Matrix<double>> m_weights;
};

#endif