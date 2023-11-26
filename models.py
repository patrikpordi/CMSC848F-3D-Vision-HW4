import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()


        # Define your layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Decoder
        self.fc = nn.Sequential(
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
        )
        

    def forward(self, points):
        # '''
        # points: tensor of size (B, N, 3)
        #         , where B is batch size and N is the number of points per object (N=10000 by default)
        # output: tensor of size (B, num_classes)
        # '''
        # pass
        
        points = points.transpose(1, 2)

        #Encoder
        features = self.conv1(points)
        features = self.bn1(features)
        features = F.relu(features)

        features = self.conv2(features)
        features = self.bn2(features)
        features = F.relu(features)

        features = self.conv3(features)
        features = self.bn3(features)
        features = F.relu(features)

        features = self.conv4(features)
        features = self.bn4(features)
        features = F.relu(features)

        features = self.conv5(features)
        features = self.bn5(features)
        features = F.relu(features)

        #Decoder
        # features = nn.MaxPool1d(features.size(-1))(features)
        features = torch.amax(features, dim=-1)

        # features = features.squeeze(-1)
        features = self.fc(features)
        return features



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()

        # Define your layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Decoder
        self.fc = nn.Sequential(
            
            nn.Conv1d(1088, 512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,num_seg_classes,1),
        )
        # pass

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''

         # Transpose to match PointNet input format
        points = points.permute(0, 2, 1)
        # print(points.shape)

        # Encoder
        features_l1 = F.relu(self.bn1(self.conv1(points)))
        features_l1 = F.relu(self.bn2(self.conv2(features_l1)))
        features_l1 = F.relu(self.bn3(self.conv3(features_l1)))
        
        features_l2 = F.relu(self.bn4(self.conv4(features_l1)))
        features_l2 = F.relu(self.bn5(self.conv5(features_l2)))

        # Global max pooling
        features_m = torch.nn.MaxPool1d(points.shape[2])(features_l2)
        
        features_m =features_m.expand(points.shape[0],1024,points.shape[2])
        # print(features_m.shape)
        
        features = torch.cat((features_l1, features_m), dim=1)
        # print("Out",features.shape)


        # Decoder
        # print("Before fc:", features.shape)
        features = self.fc(features)
        # print("After fc:", features.shape)
        features = features.permute(0, 2, 1)
        return features
        # pass



