import torch
import torch.nn as nn
import numpy as np

class net(nn.Module):
    def __init__(self):
        super.__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, inputs):
        layer = self.fc1(inputs)
        layer = self.relu(layer)
        layer = self.fc2(layer)
        return layer


def train_one_step(inputs, labels, model, criterion):
    weight_p, bias_p =[], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    logit = model(inputs)
    loss = criterion(input=logit, target=labels)
    opt = torch.optim.SGD([{'params': weight_p, 'weight_decay': 1e-5},
                           {'params': bias_p, 'weight_decay': 0}],
                          lr=1e-2,
                          momentum=0.9)

    opt.zero_grad()
    loss.backword()
    opt.step()
    print('一次更新的损失为: ', loss)


if __name__ == "__main__":
    print(torch.__version__)