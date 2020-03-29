import argparse
import csv
import numpy
import time


def main():
    parser = argparse.ArgumentParser(description="Convert fer.csv to npz")
    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=numpy.inf,
        help="Limit sampling DEFAULT: inf")
    args = parser.parse_args()

    t0 = time.time()
    x_list, y_list = [], []

    with open('fer_dataset/fer.csv') as file:
        reader = csv.reader(file)
        head = next(reader)

        for (count, row) in enumerate(reader):
            y = int(row[0])
            if y not in range(3, 6):
                continue

            x_list.append(numpy.array(list([int(i) for i in row[1].split()])))
            y_list.append(y - 3)

    t1 = time.time()
    print("Processed! {0}s".format(t1 - t0))

    value = int(len(x_list) * .2)
    train = min(int(len(x_list) * .8), args.samples)

    (x_train, x_test) = (numpy.array(
        x_list[:train]), numpy.array(x_list[-value:]))
    (y_train, y_test) = (numpy.array(
        y_list[:train]), numpy.array(y_list[-value:]))

    numpy.savez_compressed('fer_numpy/train', x_list=x_train, y_list=y_train)
    print("Training results computed | x: {0}, y: {1}".format(
        str(x_train.shape), str(y_train.shape)))

    numpy.savez_compressed('fer_numpy/test', x_list=x_test, y_list=y_test)
    print('Test results computed | x: {0}, y: {1}'.format(
        str(x_test.shape), str(y_test.shape)))

    print("Convert complete! {0}s".format(time.time() - t1))


if __name__ == '__main__':
    main()
