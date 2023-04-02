import model
import dataset
import random


def main():
    data = dataset.main("data/export20230402-102556.sqlite")
    random.seed(42)
    random.shuffle(data)
    test_data, train_data = data[:500], data[500:]
    # model.pipeline(train_data)

    # cross validation
    k = 5
    random.seed()
    random.shuffle(train_data)
    splits = [[] for _ in range(k)]

    for i in range(len(train_data)):
        splits[i % k].append(train_data[i])

    for i in range(k):
        split_test = splits[i]
        split_train = []
        for j in range(k):
            if j == i:
                continue
            split_train += splits[j]
        print(f"\n====== Split {i + 1} ======")
        model.pipeline(split_train, split_test)


if __name__ == "__main__":
    main()
