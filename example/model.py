from torch import nn


class Model(nn.Module):
    def __init__(self, inputNode, hidden, outputNode):
        super().__init__()
        self.liner_1 = nn.Linear(inputNode, hidden)
        self.liner_2 = nn.Linear(hidden, hidden)
        self.drop_layer = nn.Dropout(p=0.25)
        self.liner_2_1 = nn.Linear(hidden, hidden // 2)
        self.liner_3 = nn.Linear(hidden // 2, outputNode)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        前向传播
        """
        x = self.liner_1(input)
        x = self.relu(x)
        x = self.drop_layer(x)
        x = self.liner_2(x)
        x = self.relu(x)
        x = self.drop_layer(x)
        x = self.liner_2_1(x)
        x = self.relu(x)
        x = self.drop_layer(x)
        x = self.liner_3(x)
        x = self.sigmoid(x)
        return x
