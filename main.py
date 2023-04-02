import model
import dataset
import random


def main():
    data = dataset.main("data/export20230402-102556.sqlite")
    random.shuffle(data)
    test_data, train_data = data[:500], data[500:]
    model.pipeline(train_data)


if __name__ == "__main__":
    main()
