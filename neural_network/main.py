

def change_cur_directory():
    import os
    from pathlib import Path

    file = Path(__file__)
    parent = file.parent
    os.chdir(parent)


change_cur_directory()


if __name__ == '__main__':
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(training_data[0])

    import network
    net = network.Network([784, 30, 10])

    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
