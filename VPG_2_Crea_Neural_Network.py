import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import os

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[32]):
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

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False, output_folder='****'):
    # make environment, check spaces, get obs / act dims
    # env = gym.make(env_name, render_mode="human")
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    # print(obs_dim)
    n_acts = env.action_space.n
    # print(n_acts)

    # make core of policy network
    model = MLP(input_dim=obs_dim, output_dim=n_acts, hidden_sizes=hidden_sizes)

    # Restringir el tamaño del batch para evitar problemas de memoria
    batch_size = min(batch_size, env._max_episode_steps)

    # make function to compute action distribution
    def get_policy(obs):
        logits = model(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            if isinstance(obs,tuple):
                obs= obs[0]
            batch_obs.append(obs)

            # act in the environment
            # print("Observación", obs)
            act = get_action(torch.as_tensor(obs[0] if isinstance(obs, tuple) else obs, dtype=torch.float32))
            # print(env.step(act))
            obs, rew, done, _, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break
        

        batch_obs = torch.tensor(batch_obs)
        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=batch_obs,
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens
    
    # carpeta de salida
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, "vpg2_CartPole_12.txt")
    # training loop
    with open(file_path, 'a') as file:
        print('\nUsando la formulación más simple de gradiente de política.\n', file=file)
        # bucle de entrenamiento
        for i in range(epochs):
            batch_loss, batch_rets, batch_lens = train_one_epoch()
            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                    (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)), file=file)
            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
                    (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
            print(f"Los resultados se están guardando en: {file_path}")
    
    # Guardar el modelo entrenado
    torch.save(model.state_dict(), os.path.join(output_folder, "model_2_LunarLander_20_vpg2.pt"))

    return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--lr', type=float, default=1e-2)
    #Ubicación donde quieres que se creé tanto tu modelo entrenado, como un archivo con cada entrenamiento realizado
    parser.add_argument('--output_folder', type=str, default='****')
    parser.add_argument('--epochs', type=int, default=50)  # Número de épocas
    args = parser.parse_args()
    
    print('\nUsing reward-to-go formulation of policy gradient.\n')
    train(env_name=args.env_name, render= args, lr=args.lr,
          epochs=args.epochs, output_folder=args.output_folder)