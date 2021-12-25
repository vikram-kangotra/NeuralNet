#include "matrix.hpp"
#include "mnist.hpp"
#include "dnn.hpp"

void train(DNN& neural, const std::string& fileName, const int& count, const int& epoch) {
    Mnist mnist(fileName);

    std::vector<double> target(10);

    for (int i = 1; i <= epoch; ++i) {
        mnist.reset();
        for (int j = 1; j <= count; ++j) {
            auto data = mnist.getNextData();
            target[data.label] = 0.99;
            neural.train(data.pixels, target);
            target[data.label] = 0.01;
        }
        system("clear");
        std::cout << "Epoch " << i << " of " << epoch << '\n';
        std::cout << "Error: " << neural.getError() << std::flush;
    }
}

auto test(DNN& neural, const std::string& fileName, const int& count) {

    Mnist mnist(fileName);

    auto success = 0;

    for (int j = 1; j <= count; ++j) {
        auto data = mnist.getNextData();
        auto pred = neural.query(data.pixels);
        double max = 0;
        auto prediction = 0;
        for (int i = 0; i < pred.getRows(); ++i) {
            if (max < pred[i][0]) {
                max = pred[i][0];
                prediction = i;
            }
        }
        if (prediction == data.label) {
            ++success;
        }
        system("clear");
        std::cout << "Tested: " << j << '\n';
        std::cout << "Success: " << 100 * (success / double(j)) << "%" << std::flush;
    }
    return 100 * (success / double(count));
}

void learn(const double& percentage, const int& count, const int& epoch)
{
    srand(time(0));

    for (;;) {
        DNN neural({ 784,100,10 }, 0.1);

        train(neural, "/usr/share/mnist/mnist_train.csv", count, rand() % epoch + 1);
        auto success = test(neural, "/usr/share/mnist/mnist_test.csv", count);

        if (success > percentage) {
            neural.saveModel(std::to_string(int(success)) + "_mnist_" + std::to_string(count) + ".rwm");
            break;
        }
    }
}

void reverse_test(DNN& neural, const int& number) 
{
    std::vector<double> target(10);
    for(int i=0; i<10; ++i) {
        target[i] = 0.01;
    }
    target[number] = 0.99;
    auto output = neural.reverse_query(target);

    MnistData mnistData;
    
    for(int i=0; i<output.getRows(); ++i) {
        mnistData.pixels.push_back(output[i][0]);
    }

    Mnist mnist(mnistData);
    mnist.draw();
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
    	std::cout << "Not enough arguments were provided";
	return -1;
    }

    const auto percentage = atoi(argv[1]);
    const auto count = atoi(argv[2]);
    const auto epoch = atoi(argv[3]);
    learn(percentage, count, epoch);
    
    DNN neural = DnnModel::loadModel("83_mnist_1000.rwm");
    test(neural, "/usr/share/mnist/mnist_test.csv", 100);

    for(int i=0; i<10; ++i) {
        system("clear");
        std::cout << i << std::flush;
        reverse_test(neural, i);
    }

    return 0;
}
