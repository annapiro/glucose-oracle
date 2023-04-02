import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import device
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import Datapoint

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # for now

# hyper-parameters
EPOCHS = 20
BATCH_SIZE = 16
LR = 0.01
HIDDEN_SIZE = 12
MARGIN = 0.5  # error margin within which the prediction is considered correct


class Net(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out


class GlucoseDataset(Dataset):
    def __init__(self, points: list[Datapoint]):
        # self.features = F.normalize(feats, dim=0)
        self.features, self.targets = preprocess_dps(points)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def preprocess_dps(points: list[Datapoint]) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts features and targets from a list of Datapoint objects

    :param points: list containing Datapoint objects
    :return: tuple of preprocessed features and targets
    """
    return torch.Tensor([[x.current_bg,
                          x.average_bg,
                          x.last_carbs,
                          x.last_carbs_time,
                          x.last_insulin,
                          x.last_insulin_time] for x in points]).to(torch.device(device)), \
        torch.Tensor([x.next_bg for x in points]).to(torch.device(device))


def training(model: Net, loader: DataLoader):
    """Training loop for the model

    :param model: TODO
    :param loader:
    :return:
    """
    model.train()  # set mode to training
    loss_func = nn.MSELoss()  # define loss function
    optim = torch.optim.Adam(model.parameters(), lr=LR)  # define optimizer
    losses = []
    accs = []

    # one epoch = one pass through all training data
    for epoch in tqdm(range(EPOCHS)):
        epoch_losses = []
        epoch_accs = []

        # one batch = randomly selected subset of the training data,
        # after which the weights are updated
        for batch in loader:
            inputs, target = batch
            # clear gradients from previous epoch
            optim.zero_grad()
            # make predictions for the current batch
            pred = model(inputs).squeeze()
            # compute the loss and its gradients
            loss = loss_func(target, pred)
            loss.backward()
            # update the weights
            optim.step()

            # reporting
            epoch_losses.append(loss.item())
            batch_accs = list(map(eval_pred, zip(target, pred)))
            epoch_accs.append(sum(batch_accs) / len(batch_accs))

        losses.append(round(sum(epoch_losses) / len(epoch_losses), 2))
        accs.append(round(sum(epoch_accs) / len(epoch_accs), 2))

    print(f"Epoch losses:\n{losses}")
    print(f"Epoch accuracies:\n{accs}")


def testing(model: Net, test_set: list[Datapoint]):
    """TODO

    :param model:
    :param test_set:
    :return:
    """
    model.eval()
    feats, targets = preprocess_dps(test_set)
    with torch.no_grad():
        pred = model(feats).squeeze()
    verdicts = list(map(eval_pred, zip(targets, pred)))
    errors = sorted([abs(targets[i] - pred[i]).item() for i in range(len(pred))])
    print(f"Accuracy: {sum(verdicts) / len(verdicts):.2f}\n"
          f"Average error: {sum(errors) / len(errors):.2f}\n"
          f"Median error: {errors[len(errors) // 2]:.2f}")


def eval_pred(y_pair: tuple[float, float]) -> float:
    """Evaluates whether the prediction is correct

    :param y_pair: Tuple of target and predicted values
    :return: 1.0 if the prediction is correct, else 0.0
    """
    return 1.0 if (abs(y_pair[0] - y_pair[1]) <= MARGIN) else 0.0


def pipeline(train_data: list[Datapoint], test_data: list[Datapoint]=None):
    model = Net(6, HIDDEN_SIZE, 1)
    model = model.to(device)
    train_data = GlucoseDataset(train_data)
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    training(model, loader)

    if test_data:
        testing(model, test_data)


if __name__ == '__main__':
    print('[current bg, avg bg, last carbs, time since last carbs, last insulin, time since last insulin]')

