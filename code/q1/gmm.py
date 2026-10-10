import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import os
import numpy as np
import matplotlib.pyplot as plt
from models import GMM

if __name__ == '__main__':
    os.makedirs('outputs/gmm_bmm',exist_ok=True)
    with np.load('data/gmm_bmm/mnist_gmm.npz') as data:
        mnist_data = data
        # print(mnist_data.files)
        # print(mnist_data['X'])
        # print(mnist_data['y'])
        X = mnist_data['X']; y = mnist_data['y']
        # print(X.shape, y.shape)
    
    # _, cnts = np.unique(y,return_counts=True)
    # n_classes = len(cnts)
    n_classes = 10
    n_features = X.shape[1]
    # X_0 = X[(y==3)&(y==4)]
    # i'll probably experiment with the #components a lil bit
    gmm_per_digit = [GMM(n_components=10) for _ in range(n_classes)]
    # gmm.fit(X_0,max_steps=10)
    for k in range(n_classes):
        X_k = X[y==k]
        print(f'Training on digit {k} images')
        gmm_per_digit[k].fit(X_k,max_steps=10)

    out = []
    for k in range(n_classes):
        out.append([])
        # generate #size images
        for _ in range(5):
            # sample from 5 components
            z = np.random.choice(np.arange(10),p=gmm_per_digit[k].mixing_coefs)
            # assert False, z
            gen_img = gmm_per_digit[k].generate(z=z,size=1)
            # assert False, gen_imgs.shape
            out[k].append(gen_img)
    # assert False, out[k]
    # output generated images
    fig, ax = plt.subplots(nrows=10,ncols=5,sharey='row',figsize=(10,8))
    for k in range(n_classes):
        for i in range(len(out[k])):
            img = out[k][i].reshape(28,28)
            # assert False, img.shape
            ax[k,i].matshow(img,cmap='gray')
            ax[k,i].set_ylabel(k)
            ax[k,i].set_xticks([])
            ax[k,i].set_yticks([])
    fig.tight_layout()
    fig.savefig('outputs/gmm_bmm/gmm.png',bbox_inches='tight')
    plt.close(fig)
            