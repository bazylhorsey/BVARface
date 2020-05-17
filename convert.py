import csv
import numpy

def main():
    x_list, y_list = [], []

    with open("bvarface/fer_dataset/fer.csv") as file:
        reader = csv.reader(file)
        head = next(reader)

        for (_, row) in enumerate(reader):
            
            y = int(row[0])
            if y == 3:
                emo_val = 0
            if y == 4:
                emo_val = 1
            if y == 5:
                emo_val = 2
            else:
                continue

            x_list.append(numpy.array(list([int(i) for i in row[1].split()])))
            y_list.append(emo_val)

    print("Processed!")

    value = int(len(x_list) * .2)
    train = int(len(x_list) * .8)

    (x_train, x_test) = \
        (numpy.array(x_list[:train]), numpy.array(x_list[-value:]))
    (y_train, y_test) = \
        (numpy.array(y_list[:train]), numpy.array(y_list[-value:]))

    numpy.savez_compressed("bvarface/fer_numpy/train", x_list=x_train, y_list=y_train)

    numpy.savez_compressed("bvarface/fer_numpy/test", x_list=x_test, y_list=y_test)

    print("Convert complete!")


if __name__ == "__main__":
    main()
