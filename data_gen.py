import logging
import numpy as np
import matplotlib.pyplot as plt


def oscillating_func(x):
    y = 1000 * np.sin(0.002 * x**2) / x + 0.2 * x * np.log2(x) + 10
    return y


def gen_noisy_func_data(orig_func):
    # Create a sample domain in the range [a, b)
    a = 0
    b = 200
    sample_size = 100
    sample_x = np.round(np.sort((b - a) * np.random.random_sample(size=sample_size) + a), decimals=2)

    # Get original y values
    sample_orig_y = np.asarray([orig_func(x) for x in sample_x])

    # Create noisy data
    # Draw a sample from the normal distribution centered at each y value scaled by a specific std
    std = 10
    sample_noisy_y = np.asarray([np.random.normal(loc=y, scale=std) for y in sample_orig_y])

    return sample_x, sample_orig_y, sample_noisy_y


def plot_data(sample_x, sample_orig_y, sample_noisy_y, orig_func):
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(figsize=(15, 15))

    ax.set_title('Nondeterministic Data Generation', fontsize=40)
    if sample_orig_y is not None:
        x_ticks = np.arange(0, 200, 0.1)
        ax.plot(x_ticks, orig_func(x_ticks))
        ax.plot(sample_x, sample_orig_y, 'ob', label=r'$g$')
    if sample_noisy_y is not None:
        ax.plot(sample_x, sample_noisy_y, 'or', label=r'$\hat{g}$')
    ax.legend(loc='upper left', fontsize='40')

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    sample_x, sample_orig_y, sample_noisy_y = gen_noisy_func_data(oscillating_func)
    plot_data(sample_x, sample_orig_y, sample_noisy_y, oscillating_func)
