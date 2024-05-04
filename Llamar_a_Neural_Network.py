import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import os
import queue
import threading

# Definir la arquitectura de la red neuronal
class YourModelClass(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[32]):
        super(YourModelClass, self).__init__()
        layer_sizes = [input_dim] + hidden_sizes + [output_dim]
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())  # Capas ocultas con activación Tanh
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

ruta_modelo_entrenado = 'C:\\Users\\chema\\OneDrive\\Escritorio\\TFG\\Final\\32_neuronas\\vpg1\\model_Acrobot_1_vpg1.pt'
trained_model= torch.load(ruta_modelo_entrenado, map_location=torch.device('cpu'))

# Crear una instancia de la red neuronal
env_name='Acrobot-v1'
env = gym.make(env_name)
input_dim = env.observation_space.shape[0]  # Dimensión de entrada de tu modelo
print(input_dim)
output_dim = env.action_space.n  # Dimensión de salida de tu modelo
print(output_dim)
model = YourModelClass(input_dim, output_dim)

model.load_state_dict(trained_model)
model.eval()

def test_trained_model(trained_model, env_name='Acrobot-v1', render=False):
    # Si quieres ver como se ejecuta mientras aprendes , render_mode="human"
    env = gym.make(env_name)
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        if render:
            env.render()

        # Obtener la acción del modelo entrenado
        if isinstance(obs,tuple):
                obs= obs[0]
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_probs = torch.softmax(trained_model(obs_tensor), dim=-1)
        action = torch.argmax(action_probs, dim=-1).item()

        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        # print(total_reward)
        #He añadido esta variable de 10000, porque en VPG 3 nunca para
        if(total_reward >= 10000):
            break
            return total_reward

    env.close()
     # Guardar los resultados en el archivo
    os.makedirs('C:\\Users\\chema\\OneDrive\\Escritorio\\TFG\\Final', exist_ok=True)
    file_path = os.path.join('C:\\Users\\chema\\OneDrive\\Escritorio\\TFG\\Final\\Desviacion tipica', "model_Acrobot_1_vpg1.txt")

    with open(file_path, 'a') as file:
        print('Recompensa total: %3d' %
                      (total_reward), file=file)
    return total_reward

reward = test_trained_model(model, env_name='Acrobot-v1', render=True)
print("Reward: ", reward)