from __future__ import print_function, division
from keras.datasets import mnist
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input,Dense,Reshape,Flatten,Dropout,Concatenate
from keras.layers import BatchNormalization,Activation,ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D,Conv2D
from keras.models import Sequential,Model
from keras.optimizers import Adam
from glob import glob 
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np 
import os 
import scipy
import scipy.misc
## condation is the second half of the image.
## we generate fake image on condation image
## img_B => condation 
## img_A => real image
## Discriminator(img_A,img_B) => Real
## Discriminator(fake_A,img_B) => Fake
## fake_A = Generator(img_B) generating fake images on condation images.
class DataLoader():
    def __init__(self,dataset_name,img_res=(128,128)):
        self.dataset_name = dataset_name
        self.img_res = img_res 
    def load_data(self,batch_size=1,is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('./datasets/%s/%s/*'%(self.dataset_name,data_type))
        batch_images = np.random.choice(path,size=batch_size)
        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = self.imread(img_path)
            h,w,_ = img.shape
            _w = int(w/2)
            img_A,img_B = img[:,:_w,:],img[:,_w:,:]
            img_A = scipy.misc.imresize(img_A,self.img_res)
            img_B = scipy.misc.imresize(img_B,self.img_res)
            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.
        return imgs_A,imgs_B
    def load_batch(self,batch_size=1,is_testing=False):
        data_type = "train" if not is_testing else "val"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        self.n_batches = int(len(path)/batch_size)
        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A,imgs_B = [],[]
            for img in batch:
                img = self.imread(img)
                h,w,_ = img.shape
                half_w = int(w/2)
                img_A = img[:,:half_w,:]
                img_B = img[:,half_w:,:]
                img_A = scipy.misc.imresize(img_A,self.img_res)
                img_B = scipy.misc.imresize(img_B,self.img_res)
                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
                imgs_A.append(img_A)
                imgs_B.append(img_B)
            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            yield imgs_A,imgs_B 

    def imread(self,path):
        return scipy.misc.imread(path,mode='RGB').astype(np.float)


class Pix2Pix():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows,self.img_cols,self.channels)

        self.dataset_name = 'facades'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,img_res=(self.img_rows,self.img_cols))
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch,patch,1)
        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64
        
        optimizer = Adam(0.0002,0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])
        #self.discriminator.summary()
        #------
        # Construct Computational
        # Graph of Generator
        #-------

        # Build the generator, generator don't have evaluating metrics.
        self.generator = self.build_generator()
        
        #self.generator.summary()
        # Input images and their conditioning images

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A 
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A,img_B])

        self.combined = Model(inputs=[img_A,img_B],outputs=[valid,fake_A])
        # 'mse'== 'mean squared error' 'mae' == 'mean absolute error'
        self.combined.compile(loss=['mse','mae'],loss_weights=[1,100],optimizer=optimizer)
        self.combined.summary()
    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input,filters,f_size=4,bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters,kernel_size=f_size,strides=2,padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d 
        def deconv2d(layer_input,skip_input,filters,f_size=4,dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters,kernel_size=f_size,strides=1,padding='same',activation='relu')(u)
            if dropout_rate:
                u = DropOut(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u,skip_input])
            return u
        # Image input
        d0 = Input(shape=self.img_shape)
        # Downsampling
        d1 = conv2d(d0,self.gf,bn=False)
        d2 = conv2d(d1,self.gf*2)
        d3 = conv2d(d2,self.gf*4)
        d4 = conv2d(d3,self.gf*8)
        d5 = conv2d(d4,self.gf*8)
        d6 = conv2d(d5,self.gf*8)
        d7 = conv2d(d6,self.gf*8)

        # Upsampling

        u1 = deconv2d(d7,d6,self.gf*8)
        u2 = deconv2d(u1,d5,self.gf*8)
        u3 = deconv2d(u2,d4,self.gf*8)
        u4 = deconv2d(u3,d3,self.gf*4)
        u5 = deconv2d(u4,d2,self.gf*2)
        u6 = deconv2d(u5,d1,self.df)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels,kernel_size=4,strides=1,padding='same',activation='tanh')(u7)
        return Model(d0,output_img)
    def build_discriminator(self):
        def d_layer(layer_input,filters,f_size=4,bn=True):
            """Discriminator layer"""
            d = Conv2D(filters,kernel_size=f_size,strides=2,padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d 

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A,img_B])

        d1 = d_layer(combined_imgs,self.df,bn=False)
        d2 = d_layer(d1,self.df*2)
        d3 = d_layer(d2,self.df*4)
        d4 = d_layer(d3,self.df*8)

        validity = Conv2D(1,kernel_size=4,strides=1,padding='same')(d4)
        return Model([img_A,img_B],validity)

    def train(self,epochs,batch_size=1,sample_interval=50):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,)+self.disc_patch)
        fake = np.zeros((batch_size,)+self.disc_patch)

        for epoch in range(epochs):
            for batch_i,(imgs_A,imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                # _-----
                # Train Discriminator
                # ----
                # Condition on B and generate a translated version

                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A,imgs_B],valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A,imgs_B],fake)
                d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)

                # --------
                # Train Generator
                # --------

                # Train the generators 

                g_loss = self.combined.train_on_batch([imgs_A,imgs_B],[valid,imgs_A])
                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch,epochs,batch_i,self.data_loader.n_batches,d_loss[0],100*d_loss[1],g_loss[0],elapsed_time))

                # If at save interval => save generated image samples 
                """
                if batch_i % sample_interval == 0:
                    self.generator.save('generator.h5')
                    self.sample_images(epoch,batch_i)
                """
            if epoch % 20 == 0:
                self.generator.save(f'generator{epoch}.h5')
                self.sample_images(epoch,batch_i)
            
    def sample_images(self,epoch,batch_i):
        os.makedirs('images/%s' % self.dataset_name,exist_ok=True)
        r,c = 3,3
        imgs_A,imgs_B = self.data_loader.load_data(batch_size=3,is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B,fake_A,imgs_A])

        # Rescale images 0-1
        gen_imgs = 0.5 * gen_imgs + 0.5
        titles = ['Condition','Generated','Original']
        fig,axs = plt.subplots(r,c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=200,batch_size=8,sample_interval=200)
    #gan.summary()