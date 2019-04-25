import numpy as np
import matplotlib.pyplot as plt

def create_loss_figure(loss_list):
    fig, (left, right) = plt.subplots(ncols=2, figsize=(10,4))
    left.plot(np.arange(1, len(loss_list)+1), loss_list, label="loss")
    plt.xticks(np.arange(1, len(loss_list)+1))
    left.legend()

    right.plot(np.arange(5, 21), loss_list[4:], label="loss")
    plt.xticks(np.arange(5, 21))
    right.legend()
    fig.savefig("src/task2/loss.png")