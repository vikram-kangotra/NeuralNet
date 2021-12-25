/* #include "matrix.hpp"
#include "neuralNetwork.hpp"
#include <opencv4/highgui.hpp>

int main()
{
    NeuralNetwork<2,10,1> neural(1);

    for(int i=0; i<100000; ++i) {
        for(int i=0; i<=1; ++i) {
            for(int j=0; j<=1; ++j) {
                neural.train({double(i),double(j)}, {double(i^j)});
            }
        }
        if(cv::getCPUTickCount()%1000 == 0) {
            system("clear");
            std::cout << "Progress: " << i << std::endl;
        }
    }

    cv::Mat img(512,512, CV_8UC3, cv::Scalar(0));

    for(int i=0; i<512; ++i) {
        for(int j=0; j<512; ++j) {
            auto& color = img.at<cv::Vec3b>(i,j);
            color[0] = neural.query({i/512.0, j/512.0})[0][0] * 255;
            color[1] = neural.query({i/512.0, j/512.0})[0][0] * 255;
            color[2] = neural.query({i/512.0, j/512.0})[0][0] * 255;
            img.at<cv::Vec3b>(i,j) = color;
        }
    }

    cv::imshow("image", img);
    cv::waitKey(0);

    return 0;
} */

#include "matrix.hpp"
#include "mnist.hpp"
#include "neuralNetwork.hpp"

int main() {

    NeuralNetwork<784, 100, 10> neural(0.1);

    Mnist mnist("/usr/share/mnist/mnist_train.csv");

    Matrix<double, 10, 1> target(0);

    for(int i=0; i<4; ++i) {
        mnist.reset();
        for(int j=0; j<100; ++j) {
            auto data = mnist.getNextData();
            target[data.label][0] = 0.99;
            neural.train(data.pixels, target);
            target[data.label][0] = 0.01;
        }
    }

    mnist.reset();

    int success = 0;

    for(int j=1; j<=100; ++j) {
        auto data = mnist.getNextData();
        auto pred = neural.query(data.pixels);
        double max = 0;
        auto prediction = 0;
        for(int i=0; i<pred.getRows(); ++i) {
            if(max < pred[i][0]) {
                max = pred[i][0];
                prediction = i;
            }
        }
        if(prediction == data.label) {
            ++success;
        } else {
            std::cout << "prediction: " << prediction << std::flush;
            mnist.draw();
        }
        system("clear");
        std::cout << "Try: " << j << '\n';
        std::cout << "Success: " << success << std::flush;
    }

    return 0;
}
