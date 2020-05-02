import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

%matplotlib notebook
n = 100
# generate 4 random variables from the random, gamma, exponential, and uniform distributions
x1 = np.random.normal(-2.5, 1, n)
x2 = np.random.gamma(2, 1.5, n)
x3 = np.random.exponential(2, n)+7
x4 = np.random.uniform(14,20, n)

def update(curr):
    # check if animation is at the last frame, and if so, stop the animation a
    if curr == n:
        a.event_source.stop()
    plt.cla()
    bins = np.arange(-7, 21, 1)
    ax1.hist(x1[:curr], bins=bins, color='blue', alpha=0.5)
    ax1.axis([-7, 21, 0, 40])
    ax1.set_title('Normal')
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Value')

    ax2.hist(x2[:curr], bins=bins, color='red', alpha=0.5)
    ax2.axis([-7, 21, 0, 40])
    ax2.set_title('Gamma')
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Value')

    ax3.hist(x3[:curr], bins=bins, color='green', alpha=0.5)
    ax3.axis([-7, 21, 0, 40])
    ax3.set_title('Exponential')
    ax3.set_ylabel('Frequency')
    ax3.set_xlabel('Value')

    ax4.hist(x4[:curr], bins=bins, color='orange', alpha=0.5)
    ax4.axis([-7, 21, 0, 40])
    ax4.set_title('Uniform')
    ax4.set_ylabel('Frequency')
    ax4.set_xlabel('Value')
    ax4.annotate('n = {}'.format(curr), [13,37])
    plt.tight_layout()


# plot the histograms
##fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
#ax1.hist(x1, normed=True, bins=20, alpha=0.5)
#ax2.hist(x2, normed=True, bins=20, alpha=0.5)
#ax3.hist(x3, normed=True, bins=20, alpha=0.5)
#ax4.hist(x4, normed=True, bins=20, alpha=0.5);
#plt.axis([-7,21,0,0.6])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 6))
a = animation.FuncAnimation(fig, update, interval=100)

#plt.text(x1.mean()-1.5, 0.5, 'x1\nNormal')
#plt.text(x2.mean()-1.5, 0.5, 'x2\nGamma')
#plt.text(x3.mean()-1.5, 0.5, 'x3\nExponential')
#plt.text(x4.mean()-1.5, 0.5, 'x4\nUniform')
