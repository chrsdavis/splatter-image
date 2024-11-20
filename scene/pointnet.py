import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        self.k = k

        self.fc1 = nn.Linear(k, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)

        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)

        iden = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1)
        x = x.view(-1, self.k, self.k) + iden.cuda()
        return x

class PointNet(nn.Module):
    def __init__(self, output_channels=14):
        super(PointNet, self).__init__()

        self.input_transform = TNet(k=6)
        self.feature_transform = TNet(k=64)

        self.conv1 = nn.Conv1d(6, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1088, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_channels)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(1)

        x = x.permute(0, 2, 1)  # Permute to [batch_size, 6, num_points]

        transform = self.input_transform(x)
        x = torch.bmm(transform, x)
        x = x.permute(0, 2, 1)  # Unpermute back to [batch_size, num_points, 6]

        x = F.relu(self.bn1(self.conv1(x)))

        # Apply feature transform
        transform_feat = self.feature_transform(x)
        x = torch.bmm(transform_feat, x)
        x = x.permute(0, 2, 1)  # Unpermute back to [batch_size, num_points, 64]

        point_features = x  # Save the nx64 feature map

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Global feature vector
        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(batch_size, -1)  # Shape: [batch_size, 1024]

        # Concatenate global feature with point features
        point_features = point_features.permute(0, 2, 1)  # Shape: [batch_size, num_points, 64]
        global_feature = x.unsqueeze(1).repeat(1, num_points, 1)  # Shape: [batch_size, num_points, 1024]
        x = torch.cat([point_features, global_feature], dim=2)  # Shape: [batch_size, num_points, 1088]

        # Flatten for fully connected layers
        x = x.view(batch_size * num_points, -1)  # Shape: [batch_size * num_points, 1088]

        # TODO: remove this
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        x = x.view(batch_size, num_points, -1)  # Shape: [batch_size, num_points, output_channels]
        return x  # Per-point output
