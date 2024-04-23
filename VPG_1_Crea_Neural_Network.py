import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import os

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[1]):
        super(MLP, self).__init__()
        layer_sizes = [input_dim] + hidden_sizes + [output_dim]
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())  # Capas ocultas con activación Tanh
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train(env_name='MountainCar-v0', hidden_sizes=[1], lr=1e-2, 
          epochs=50, batch_size=5000, render=False, output_folder='****'):

    # Crear ambiente, verificar espacios, obtener dimensiones de obs / act
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "Este ejemplo solo funciona para entornos con espacios de estado continuo."
    assert isinstance(env.action_space, Discrete), \
        "Este ejemplo solo funciona para entornos con espacios de acción discretos."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    print(obs_dim)
    print(n_acts)

    # Hacer el núcleo de la red de políticas
    model = MLP(input_dim=obs_dim, output_dim=n_acts, hidden_sizes=hidden_sizes)

    # Restringir el tamaño del batch para evitar problemas de memoria
    # batch_size = min(batch_size, env._max_episode_steps)

    # Hacer función para calcular la distribución de acciones
    def get_policy(obs):
        logits = model(obs)
        return Categorical(logits=logits)
    
    # Hacer función de selección de acciones (devuelve acciones int, muestreadas de la política)
    def get_action(obs):
        return get_policy(obs).sample().item()
    
    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # Hacer optimizador
    optimizer = Adam(model.parameters(), lr=lr)

    # Para entrenar la política
    def train_one_epoch():
        # Hacer algunas listas vacías para el registro.
        batch_obs = []          # para observaciones
        batch_acts = []         # para acciones
        batch_weights = []      # para ponderación R(tau) en gradiente de política
        batch_rets = []         # para medir retornos de episodios
        batch_lens = []         # para medir longitudes de episodios

        # Reiniciar variables específicas del episodio
        obs = env.reset()       # primera observación proviene de la distribución inicial
        done = False            # señal del entorno de que el episodio ha terminado
        ep_rews = []            # lista para recompensas acumuladas a lo largo del episodio

        # renderizar primer episodio de cada época
        finished_rendering_this_epoch = False

        # recolectar experiencia actuando en el entorno con la política actual
        while True:

            # renderizar
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # guardar obs
            if isinstance(obs,tuple):
                obs= obs[0]
            batch_obs.append(obs)

            # actuar en el entorno
            act = get_action(torch.as_tensor(obs[0] if isinstance(obs, tuple) else obs, dtype=torch.float32))
            obs, rew, done, _, _ = env.step(act)

            # guardar acción, recompensa
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # si el episodio ha terminado, registrar información sobre el episodio
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # el peso para cada logprob(a|s) es R(tau)
                batch_weights += [ep_ret] * ep_len

                # reiniciar variables específicas del episodio
                obs, done, ep_rews = env.reset(), False, []

                # no renderizar de nuevo esta época
                finished_rendering_this_epoch = True

                # terminar bucle de experiencia si tenemos suficiente
                if len(batch_obs) > batch_size:
                    break

        # realizar un solo paso de actualización de gradiente de política
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # carpeta de salida
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, "vpg1_3_Acrobot_20.txt")

    with open(file_path, 'a') as file:
        print('\nUsando la formulación más simple de gradiente de política.\n', file=file)
        # bucle de entrenamiento
        for i in range(epochs):
                batch_loss, batch_rets, batch_lens = train_one_epoch()
                print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
                      (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)), file=file)
                print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
                      (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
                print(f"Los resultados se están guardando en: {file_path}")

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), os.path.join(output_folder, "model_3_Acrobot_20_vpg1.pt"))

    return model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='Acrobot-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    #Ubicación donde quieres que se creé tanto tu modelo entrenado, como un archivo con cada entrenamiento realizado
    parser.add_argument('--output_folder', type=str, default='****')
    parser.add_argument('--epochs', type=int, default=50)  # Número de épocas
    args = parser.parse_args()

    print('\nEntrenando la red neuronal...\n')
    trained_model = train(env_name=args.env_name, render=args.render, lr=args.lr,
                          epochs=args.epochs, output_folder=args.output_folder)
