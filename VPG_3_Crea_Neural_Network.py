import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
import time
import os

# Definición de la red neuronal para la política
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_sizes=[32]):
        super(PolicyNetwork, self).__init__()
        sizes = [obs_dim] + hidden_sizes + [n_actions]
        self.model = mlp(sizes)
        
    def forward(self, obs):
        return self.model(obs)

# Definición de la red neuronal para la función de valor
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=[32]):
        super(ValueNetwork, self).__init__()
        sizes = [obs_dim] + hidden_sizes + [1]
        self.value_net = mlp(sizes)
        
    def forward(self, obs):
        return self.value_net(obs)

# Función para construir una red neuronal de varias capas
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# Función para calcular la suma de recompensas futuras
def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

# Función de entrenamiento
def train(env_name='LunarLander-v2', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False, output_folder='****'):

    # Crear el entorno
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]  # Dimensión de las observaciones
    n_acts = env.action_space.n  # Número de acciones disponibles

    # Inicializar la red neuronal para la política y la función de valor
    model = PolicyNetwork(obs_dim, n_acts, hidden_sizes)
    value_net = ValueNetwork(obs_dim, hidden_sizes)

    # Inicializar los optimizadores para la política y la función de valor con Adam
    optimizer_policy = Adam(model.parameters(), lr=lr)
    optimizer_value = Adam(value_net.parameters(), lr=lr)

    # Restringir el uso de la GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    value_net.to(device)

    # Restringir el tamaño del batch para evitar problemas de memoria
    batch_size = min(batch_size, env._max_episode_steps)

    # Función para obtener la distribución de probabilidad de las acciones
    def get_policy(obs):
        logits = model.forward(obs)
        return Categorical(logits=logits)

    # Función para obtener una acción de acuerdo a la política
    def get_action(obs):
        return get_policy(obs).sample().item()

    # Función para calcular la pérdida de la política
    def compute_policy_loss(obs, act, advantages):
        logp = get_policy(obs).log_prob(act)
        return -(logp * advantages).mean() #mean calcula el promedio

    # Función para calcular la pérdida de la función de valor
    def compute_value_loss(obs, rets):
        values = value_net.forward(obs)
        return ((values - rets) ** 2).mean()

    # Función para entrenar un episodio
    def train_one_epoch():
        start_time=time.time()
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []
        batch_advantages = []

        obs = env.reset()
        done = False
        ep_rews = []

        while True:
            if render:
                env.render()

            # Manejar casos en los que obs es una tupla
            if isinstance(obs, tuple):
                obs = obs[0]
            batch_obs.append(obs)

            # Obtener acción según la política y avanzar el entorno
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _, _ = env.step(act)
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                batch_weights += list(reward_to_go(ep_rews))

                # Calcular ventajas
                values = value_net(torch.tensor(batch_obs, dtype=torch.float32))
                values = values.squeeze()  # Aplanar para que tenga la misma forma que batch_weights
                advantages = torch.tensor(batch_weights, dtype=torch.float32) - values   # Hemos decidio hacer una Advante= R - V
                batch_advantages = advantages
                obs, done, ep_rews = env.reset(), False, []

                if len(batch_obs) > batch_size:
                    break
        
        batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
        batch_acts = torch.tensor(batch_acts, dtype=torch.int32)
        batch_advantages = torch.tensor(batch_advantages, dtype=torch.float32)

        # Actualizar la política y la función de valor
        optimizer_policy.zero_grad()
        batch_policy_loss = compute_policy_loss(obs=batch_obs,
                                                act=batch_acts,
                                                advantages=batch_advantages)
        batch_policy_loss.backward()     # Realiza el gradiente del policy_loss
        optimizer_policy.step()          # Actualiza la política

        optimizer_value.zero_grad()
        batch_value_loss = compute_value_loss(obs=batch_obs,
                                              rets=torch.tensor(batch_rets, dtype=torch.float32))
        batch_value_loss.backward()      # Reliza el gradiente del value_loss
        optimizer_value.step()           # Actualiza el value function
        end_time=time.time()
        time_difference= end_time-start_time
        return batch_policy_loss, batch_value_loss, batch_rets, batch_lens, time_difference

    # carpeta de salida
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, "vpg3_CartPole_17.txt")

    # Bucle de entrenamiento
    with open(file_path, 'a') as file:
        print('\nUsando la formulación más simple de gradiente de política.\n', file=file)
        # bucle de entrenamiento
        for i in range(epochs):
            batch_policy_loss, batch_value_loss, batch_rets, batch_lens, time_difference = train_one_epoch()
            print('epoch: %3d \t policy_loss: %.3f \t value_loss: %.3f \t return: %.3f \t ep_len: %.3f \t time: %.3f'%
                    (i, batch_policy_loss, batch_value_loss, np.mean(batch_rets), np.mean(batch_lens), time_difference), file=file)
            print('epoch: %3d \t policy_loss: %.3f \t value_loss: %.3f \t return: %.3f \t ep_len: %.3f \t time: %.3f'%
                    (i, batch_policy_loss, batch_value_loss, np.mean(batch_rets), np.mean(batch_lens), time_difference))
            print(f"Los resultados se están guardando en: {file_path}")
    # Guardar el modelo entrenado
    torch.save(model.state_dict(), os.path.join(output_folder, "model_LunarLander_20_vpg3.pt"))

    return model

# Ejecutar el código si se llama directamente
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--lr', type=float, default=1e-2)  # Tasa de aprendizaje ajustada
    parser.add_argument('--gamma', type=float, default=0.99)  # Factor de descuento ajustado
    #Ubicación donde quieres que se creé tanto tu modelo entrenado, como un archivo con cada entrenamiento realizado
    parser.add_argument('--output_folder', type=str, default='****')
    parser.add_argument('--epochs', type=int, default=50)  # Número de épocas
    args = parser.parse_args()
    print('\nUsing reward-to-go formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr,
          epochs=args.epochs, output_folder=args.output_folder)