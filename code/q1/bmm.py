import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import os
import numpy as np
import matplotlib.pyplot as plt
from models import BMM

if __name__ == '__main__':
    os.makedirs('outputs/gmm_bmm',exist_ok=True)
    with np.load('data/gmm_bmm/mnist_bmm.npz') as data:
        mnist_data = data
        X = mnist_data['X']; y = mnist_data['y']
    
    n_classes = 10
    n_features = X.shape[1]

    bmm_per_digit = [BMM(n_components=10) for _ in range(n_classes)]
    for k in range(n_classes):
        X_k = X[y==k]
        print(f'Training on digit {k} images')
        bmm_per_digit[k].fit(X_k,max_steps=10)

    out = []
    for k in range(n_classes):
        out.append([])
        # generate #size images
        for _ in range(5):
            # sample from 5 components
            # assert False, bmm_per_digit[k].mixing_coefs
            z = np.random.choice(np.arange(10),p=bmm_per_digit[k].mixing_coefs)
            gen_img = bmm_per_digit[k].generate(z=z,size=1)
            out[k].append(gen_img)

    fig, ax = plt.subplots(nrows=10,ncols=5,sharey='row',figsize=(10,8))
    for k in range(n_classes):
        for i in range(len(out[k])):
            img = out[k][i].reshape(28,28)
            ax[k,i].matshow(img,cmap='gray')
            ax[k,i].set_ylabel(k)
            ax[k,i].set_xticks([])
            ax[k,i].set_yticks([])
    fig.tight_layout()
    fig.savefig('outputs/gmm_bmm/bmm.png',bbox_inches='tight')
    plt.close(fig)
            