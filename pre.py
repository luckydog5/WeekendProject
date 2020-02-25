import keras 
import numpy as np 
import matplotlib.pyplot as plt 
from train_model import Pix2Pix
from keras.models import load_model

if __name__ == '__main__':

    pix2pix = Pix2Pix()
    imgs_A,imgs_B = pix2pix.data_loader.load_data(batch_size=3,is_testing=True)
    gan = load_model('generator100.h5')
    fake_A = gan.predict(imgs_B)
    print(f'fake_A shape {fake_A.shape}')
    gen_imgs = np.concatenate([imgs_B,fake_A,imgs_A])
    gen_imgs = 0.5 * gen_imgs + 0.5
    titles = ['condition','generated','original']
    r,c = 3,3
    fix,axs = plt.subplots(r,c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].set_title(titles[i])
            axs[i,j].axis('off')
            cnt += 1

    plt.show()
    plt.close()