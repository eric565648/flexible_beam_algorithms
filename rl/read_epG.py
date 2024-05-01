import numpy as np
from matplotlib import pyplot as plt

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

directory='results/'
episode_G = np.load(directory+'episode_G.npy')

# padding for moving average
n=10
episode_G_ma = np.append(np.repeat(episode_G[0],n/2),episode_G)
episode_G_ma = np.append(episode_G_ma,np.repeat(episode_G_ma[-1],n/2))
episode_G_ma = moving_average(episode_G_ma,n)

plt.plot(episode_G)
plt.plot(episode_G_ma)
plt.title('Training Curve')
plt.show()