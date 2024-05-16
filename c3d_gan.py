import torch
import torch.nn as nn


class C3dGenerator(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(C3dGenerator, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.conv5a_2 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b_2 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv4a_2 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b_2 = nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv3a_2 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b_2 = nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv2_2 = nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv1_2 = nn.Conv3d(64, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        x = self.relu(self.conv1(x)) #[1, 64, 16, 320, 240]=7864
        x = self.pool1(x) #[1, 64, 16, 160, 120]

        x = self.relu(self.conv2(x)) #[1, 128, 16, 160, 120]=3932

        x = self.pool2(x) #[1, 128, 8, 80, 60]

        x = self.relu(self.conv3a(x)) #[1, 256, 8, 80, 60] =983
        x = self.relu(self.conv3b(x)) #[1, 256, 8, 80, 60]

        x = self.pool3(x) #[1, 256, 4, 40, 30]

        x = self.relu(self.conv4a(x)) #[1, 512, 4, 40, 30]=245
        x = self.relu(self.conv4b(x)) #[1, 512, 4, 40, 30]
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        
        x = torch.nn.functional.interpolate(x, (2, 20, 15))
        x = self.relu(self.conv5a_2(x))
        x = self.relu(self.conv5b_2(x))
        
        x = torch.nn.functional.interpolate(x, (4, 40, 30))
        x = self.relu(self.conv4a_2(x))
        x = self.relu(self.conv4b_2(x))
        x = torch.nn.functional.interpolate(x, (8, 80, 60))
        x = self.relu(self.conv3a_2(x))
        x = self.relu(self.conv3b_2(x))
        x = torch.nn.functional.interpolate(x, (16, 160, 120))
        x = self.relu(self.conv2_2(x))
        x = torch.nn.functional.interpolate(x, (16, 320, 240))
        x = self.relu(self.conv1_2(x))  # [1, 3, 16, 320, 240]
        # print(x.shape)

        return x


class C3dDescriminator(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(C3dDescriminator, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(45056, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))

        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))

        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 45056)
        x = self.relu(self.fc6(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc7(x))
        # x = self.dropout(x)
        x = self.fc8(x)
        logits = self.sigmoid(x)
        return logits

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 320, 240)
    net = C3dGenerator(num_classes=101, pretrained=False)

    outputs = net.forward(inputs)
    # print(outputs.size())
