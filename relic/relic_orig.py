import torch.nn as nn
import torchvision

from relic.modules.resnet_hacks import modify_resnet_model
from relic.modules.identity import Identity


class ReLIC(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(ReLIC, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim, bias=False),
        )
        
        
    def forward(self, x_i, x_j, x_orig):
        

        #X_i
        #online_1= self.predictor(self.projector(self.encoder(x_i)))
        raw_1= self.encoder(x_i) 
        target_1= self.projector(raw_1)
        online_1= self.predictor(target_1)

        #X_j
        #online_2= self.predictor(self.projector(self.encoder(x_j)))
        raw_2= self.encoder(x_j)
        target_2= self.projector(raw_2)
        online_2= self.predictor(target_2)
        
        #X_orig
        original_features= self.predictor(self.projector(self.encoder(x_orig)))

        return raw_1, raw_2, online_1,online_2,target_1,target_2, original_features


        '''
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        h_orig= self.encoder(x_orig)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        z_orig= self.projector(z_orig)

        k_i= self.predictor(z_i)
        k_j= self.predictor(z_j)
        k_orig= self.predictor(z_orig)
        


        return h_i, h_j,h_orig, z_i, z_j,z_orig,k_i,k_j,k_orig
        '''
