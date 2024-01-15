import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        
        n_input_channels = observation_space.shape[0]   # for 2D input, in_channels==1  
        # print(n_input_channels) # [6, 7]
        in_channels =1
        out_channels = 1
        # TODO how to initialize the kernel? how to decide the output channels?
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=2, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        # print(th.as_tensor(observation_space.sample()))
        # print(th.as_tensor(observation_space.sample()).shape)
        # print(th.as_tensor(observation_space.sample()[None]))   # add a dimension 
        # print(th.as_tensor(observation_space.sample()[None][None]))
            
        with th.no_grad(): 
            obs_sample = th.as_tensor(observation_space.sample()[None])
            if in_channels == 1:
                obs_sample = obs_sample[None]
            n_flatten = self.cnn(obs_sample.float()).shape[1]  # n_flatten shape: [batch_dim, n]
            # print('n_flatten: ', n_flatten)     

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),   
                                    nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print(observations.shape)
        # print(observations.ndim)
        for _ in range(4-observations.ndim): 
            observations = observations.unsqueeze(0)
        # print(observations.shape)
        # print(observations.ndim)
        return self.linear(self.cnn(observations))


if __name__ == '__main__':       
    ip = th.randn(5, 1, 6, 7)
    print(ip)
    in_channels =1
    out_channels = 1
    m1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))
    m2 = nn.ReLU()
    m3 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))
    m4 = nn.ReLU()
    m5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
    m6 = nn.Flatten()
    op = m1(ip)
    print(op.size())
    op = m2(op)
    print(op.size())
    op = m3(op)
    print(op.size())
    op = m4(op)
    print(op.size())
    op = m5(op)
    print(op.size())
    op = m6(op)
    print(op.size())
    print('ooooo')
    
    cnn_model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
        nn.ReLU(),
        nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
        nn.Flatten()
    )
    cp = cnn_model(ip)
    print(cp.size())
    print(cp.shape)
    
    
    mlp_model = nn.Sequential(
        nn.Linear(cp.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 3),
        nn.Tanh()
    )
    cp2 = mlp_model(cnn_model(ip))
    print(cp2.size())
    
    
    