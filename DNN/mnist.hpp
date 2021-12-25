#ifndef MNIST_HPP
#define MNIST_HPP

#include <opencv4/imgproc.hpp>
#include <opencv4/imgcodecs.hpp>
#include <opencv4/highgui.hpp>
#include <fstream>
#include <sstream>

struct MnistData {
    int label;
    std::vector<double> pixels;
};

class Mnist {
public:
    Mnist(const std::string& fileName)
    :   m_file(fileName) {
        m_file.exceptions(std::ios::failbit);

        if(!m_file.is_open()) {
            throw std::invalid_argument(fileName + " not available");
        }
    }

    Mnist(const MnistData& mnistData)
    :   m_mnistData(mnistData) {}

    bool hasData() const {
        return !m_file.eof();
    }

    const auto getNextData() {

        std::string line;
        m_file >> line;

        char skip;
        int col;

        std::stringstream ss(line);
        ss >> col >> skip;

        m_mnistData.label = col;

        m_mnistData.pixels.clear();

        for(int i=0; i<28*28; ++i) {
            ss >> col >> skip;
            m_mnistData.pixels.push_back((col/255.0 * 0.99) + 0.01);
        }
        
        return m_mnistData;   
    }

    void draw() {
        cv::Mat img(28, 28, CV_8UC3, cv::Scalar(0));

        for(int i=0; i<28; ++i) {
            for(int j=0; j<28; ++j) {
                auto pixel = img.at<cv::Vec3b>(j,i);
                pixel[0] = m_mnistData.pixels[i*28 + j + 1] * 255;
                pixel[1] = m_mnistData.pixels[i*28 + j + 1] * 255;
                pixel[2] = m_mnistData.pixels[i*28 + j + 1] * 255;
                img.at<cv::Vec3b>(j,i) = pixel;
            }
        }

        cv::flip(img, img, 0);
        cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);

        cv::resize(img, img, cv::Size(), 10, 10);

        cv::imshow("image", img);
        cv::waitKey(0);
    }

    void reset() {
        m_file.seekg(0, std::ios::beg);
    }
private:
    std::ifstream m_file;
    MnistData m_mnistData;
};

#endif