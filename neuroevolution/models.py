"""Neural network architectures for neuroevolution."""

import torch
import torch.nn as nn
import numpy as np


class CartPoleNet(nn.Module):
    """PyTorch Neural Network for CartPole, LunarLander, MountainCar, Maze, FourRooms, and CarRacing environments."""
    
    def __init__(self, input_size: int, hidden_sizes: list[int] | tuple[int, ...], output_size: int):
        super(CartPoleNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Store architecture info
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return self.network(x)
    
    def get_action(self, state, env_type: str = 'CartPole'):
        """Get action based on network output and environment type."""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
                if state.dim() in (1, 3):
                    state = state.unsqueeze(0)
            elif isinstance(state, (list, tuple)):
                state = torch.FloatTensor(np.array(state))
                if state.dim() in (1, 3):
                    state = state.unsqueeze(0)
            
            if state.dim() > 2:
                state = torch.flatten(state, start_dim=1)
            
            output = self.forward(state)
            
            if env_type == 'CartPole':
                # Binary decision: left (0) or right (1)
                action = 1 if output.item() > 0.0 else 0
            elif env_type == 'MountainCar':
                # Three discrete actions: left (0), no action (1), right (2)
                if self.output_size == 3:
                    action = torch.argmax(output, dim=1).item()
                elif self.output_size == 1:
                    # Bang-bang control: positive value pushes right (2), non-positive pushes left (0)
                    action = 2 if output.item() > 0.0 else 0
                else:
                    action = torch.argmax(output, dim=1).item()
            elif env_type == 'LunarLander':
                # Four discrete actions: do nothing (0), fire left (1), fire main (2), fire right (3)
                action = torch.argmax(output, dim=1).item()
            elif env_type in ('Maze', 'HardMaze'):
                # HardMaze: 3 continuous motor actions bounded in [0.0, 1.0]
                if self.output_size == 3:
                    action = torch.sigmoid(output).squeeze(0).cpu().numpy().astype(np.float32)
                else:
                    action = torch.argmax(output, dim=1).item()
            elif env_type in ('CarRacing', 'CarRacing-v0', 'CarRacing-v2', 'CarRacing-v3'):
                # CarRacing: continuous action [steering (-1 to 1), gas (0 to 1), brake (0 to 1)]
                if self.output_size == 3:
                    steering = torch.tanh(output[:, 0]).item()
                    gas_raw = torch.sigmoid(output[:, 1]).item()
                    brake_raw = torch.sigmoid(output[:, 2]).item()
                    if gas_raw >= brake_raw:
                        gas = gas_raw
                        brake = 0.0
                    else:
                        gas = 0.0
                        brake = brake_raw * 0.6
                    action = np.array([steering, gas, brake], dtype=np.float32)
                elif self.output_size == 5:
                    action = torch.argmax(output, dim=1).item()
                else:
                    action = output.squeeze(0).cpu().numpy().astype(np.float32)
            elif env_type in ('FourRooms', 'MiniGrid', 'MiniGrid-FourRooms-v0'):
                # MiniGrid: discrete actions selected via argmax
                action = torch.argmax(output, dim=1).item()
            else:
                # Default behavior
                if self.output_size > 1:
                    action = torch.argmax(output, dim=1).item()
                else:
                    action = 1 if output.item() > 0.0 else 0
                    
            return action
    
    def get_weights_as_vector(self) -> torch.Tensor:
        """Get all network parameters as a single vector."""
        return torch.cat([param.data.view(-1) for param in self.parameters()])
    
    def set_weights_from_vector(self, weights_vector) -> None:
        """Set network parameters from a vector."""
        if isinstance(weights_vector, np.ndarray):
            weights_vector = torch.FloatTensor(weights_vector)
            
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param.data = weights_vector[idx:idx + param_size].view(param.shape)
            idx += param_size
    
    def get_total_params(self) -> int:
        """Get total number of parameters in the network."""
        return sum(p.numel() for p in self.parameters())


# Alias for general naming
NeuralNetwork = CartPoleNet
FeedForwardNet = CartPoleNet
