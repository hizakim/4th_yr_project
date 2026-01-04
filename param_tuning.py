from tkinter import N
import matplotlib.pyplot as plt
import numpy as np

# Set larger font sizes globally
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})

#Heatmap 1
batch_size = np.array([32, 64, 128, 256])
epochs = np.array([5,10])

mean_reward = np.array([4.2442,5.1626,4.3823,3.6015,4.1667,5.2288,4.4338,4.6762])
mean_reward = mean_reward.reshape(len(epochs), len(batch_size))

plt.figure(figsize=(7, 5))
im = plt.imshow(mean_reward, cmap='bwr', origin='lower', aspect='auto')
cbar = plt.colorbar(im, label='Mean reward')
cbar.set_label('Mean reward', fontsize=18)

plt.xticks(range(len(batch_size)), batch_size, fontsize=16)
plt.yticks(range(len(epochs)), epochs, fontsize=16)
'''
# add reward values as text
for i in range(len(epochs)):
    for j in range(len(batch_size)):
        plt.text(j, i, f'{mean_reward[i, j]:.4f}', ha='center', va='center', color='black', fontsize=16)
'''
plt.xlabel('Batch size', fontsize=18)
plt.ylabel('Epochs', fontsize=18)
plt.tight_layout(pad=1.5)
plt.show()

##########################

#Heatmap 2
#batch_size = 64

epochs = np.array([3,5,7])
n_steps = np.array([250,500,750])

mean_reward = np.array([3.6827,4.3034,4.3866,4.5551,5.2288,3.5316,4.148,3.6346,3.4773])
mean_reward = mean_reward.reshape(len(n_steps), len(epochs))

plt.figure(figsize=(7, 5))
im = plt.imshow(mean_reward, cmap='bwr', origin='lower', aspect='auto')
cbar = plt.colorbar(im, label='Mean reward')
cbar.set_label('Mean reward', fontsize=18)

plt.xticks(range(len(epochs)), epochs, fontsize=16)
plt.yticks(range(len(n_steps)), n_steps, fontsize=16)

'''
# add reward values as text
for i in range(len(n_steps)):
    for j in range(len(epochs)):
        plt.text(j, i, f'{mean_reward[i, j]:.4f}', ha='center', va='center', color='black', fontsize=16)
'''
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('n_steps', fontsize=18)
plt.tight_layout(pad=1.5)
plt.show()

##############################
#batch_size = 64
#epochs = 5
#n_steps = 500

clipping_range = np.array([0.05,0.1,0.2,0.3])
mean_reward_clipping = np.array([4.5794,5.2288,5.1625,4.6603])
std_clipping = np.array([1.2768,1.2344,1.5916,1.3689])

learning_rate = np.array([0.03,0.003,0.001,0.0003,0.0001,0.00003])
mean_reward_learning_rate = np.array([3.6291,3.8205,4.4177,5.2288,4.4848,3.4372])
std_learning_rate = np.array([1.5192, 1.3655, 1.1338, 1.2344, 1.4877, 1.836])

gamma = np.array([0.9,0.95,0.99,0.9925,0.995, 0.999])
mean_reward_gamma = np.array([4.8070,4.7635,4.7696,4.9152,5.2288,5.1071])
std_gamma = np.array([1.8794,1.3801,1.5399,1.3472,1.2344,1.4075])

entropy_coef = np.array([0.01,0.015,0.02,0.025,0.03])
mean_reward_entropy_coef = np.array([5.0141,5.0239,5.2288,4.7486,4.9929])
std_entropy_coef = np.array([1.5926,1.4278, 1.2344,1.6252, 1.3836])

value_coef = np.array([0.4, 0.5, 0.6,0.75])
mean_reward_value_coef = np.array([4.566, 5.2288, 5.0841,5.0179])
std_value_coef = np.array([1.5473,1.2344,1.1681,1.4187])

gae_lambda = np.array([0.85,0.9,0.95,0.99])
mean_reward_gae_lambda = np.array([3.7876,4.1744,5.2288,4.0447])
std_gae_lambda = np.array([1.5939,1.7863,1.2344, 1.1915])

# Clipping range
plt.figure(figsize=(6, 5))
plt.fill_between(clipping_range, mean_reward_clipping - std_clipping, mean_reward_clipping + std_clipping,
                 alpha=0.3, color='blue', label='±1 std')
plt.plot(clipping_range, mean_reward_clipping, 'o-', color='blue', markersize=8)
plt.xlabel('Clipping range', fontsize=18)
plt.ylabel('Mean reward', fontsize=18)
plt.tick_params(labelsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tight_layout(pad=1.5)
plt.show()

# Learning Rate
plt.figure(figsize=(6, 5))
plt.fill_between(learning_rate, mean_reward_learning_rate - std_learning_rate, mean_reward_learning_rate + std_learning_rate,
                 alpha=0.3, color='green', label='±1 std')
plt.plot(learning_rate, mean_reward_learning_rate, 'o-', color='green', markersize=8)
plt.xlabel('Learning rate', fontsize=18)
plt.ylabel('Mean reward', fontsize=18)
plt.tick_params(labelsize=16)
plt.grid(True)
plt.xscale('log')
plt.legend(fontsize=16)
plt.tight_layout(pad=1.5)
plt.show()

# Gamma
plt.figure(figsize=(6, 5))
plt.fill_between(gamma, mean_reward_gamma - std_gamma, mean_reward_gamma + std_gamma,
                 alpha=0.3, color='red', label='±1 std')
plt.plot(gamma, mean_reward_gamma, 'o-', color='red', markersize=8)
plt.xlabel('Gamma', fontsize=18)
plt.ylabel('Mean reward', fontsize=18)
plt.tick_params(labelsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tight_layout(pad=1.5)
plt.show()

# Entropy Coef
plt.figure(figsize=(6, 5))
plt.fill_between(entropy_coef, mean_reward_entropy_coef - std_entropy_coef, mean_reward_entropy_coef + std_entropy_coef,
                 alpha=0.3, color='purple', label='±1 std')
plt.plot(entropy_coef, mean_reward_entropy_coef, 'o-', color='purple', markersize=8)
plt.xlabel('Entropy coef', fontsize=18)
plt.ylabel('Mean reward', fontsize=18)
plt.tick_params(labelsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tight_layout(pad=1.5)
plt.show()

# Value Coef
plt.figure(figsize=(6, 5))
plt.fill_between(value_coef, mean_reward_value_coef - std_value_coef, mean_reward_value_coef + std_value_coef,
                alpha=0.3, color='orange', label='±1 std')
plt.plot(value_coef, mean_reward_value_coef, 'o-', color='orange', markersize=8)
plt.xlabel('Value coef', fontsize=18)
plt.ylabel('Mean reward', fontsize=18)
plt.tick_params(labelsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tight_layout(pad=1.5)
plt.show()

# GAE Lambda
plt.figure(figsize=(6, 5))
plt.fill_between(gae_lambda, mean_reward_gae_lambda - std_gae_lambda, mean_reward_gae_lambda + std_gae_lambda,
                alpha=0.3, color='brown', label='±1 std')
plt.plot(gae_lambda, mean_reward_gae_lambda, 'o-', color='brown', markersize=8)
plt.xlabel('GAE lambda', fontsize=18)
plt.ylabel('Mean reward', fontsize=18)
plt.tick_params(labelsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tight_layout(pad=1.5)
plt.show()