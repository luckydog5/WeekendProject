import numpy as np 
import matplotlib.pyplot as plt 
import keys
import os 
import multiprocessing
from PIL import Image
from crnn2 import get_model


class Predict(object):
    def __init__(self,modelpath):
        
        self.modelpath = modelpath
        self.characters = self.load_characters()
        self.model = self.load_model()
    def load_characters(self):
        characters = keys.alphabet[:]
        characters = characters[1:] + u'卍'
        return characters
    def load_model(self):
        model = get_model(False,32,len(self.characters))
        model.load_weights(self.modelpath)
        return model 
    def decode(self,pred):
        char_list = []
        pred_text = pred.argmax(axis=2)[0]
        for i in range(len(pred_text)):
            if pred_text[i] != len(self.characters) -1 and ((not(i>0 and pred_text[i]==pred_text[i-1])) or (i>1 and pred_text[i]==pred_text[i-2])):
                char_list.append(self.characters[pred_text[i]])
        return u''.join(char_list)
    def predict(self,image):
        width,height = image.size[0],image.size[1]
        scale = height * 1.0 / 32
        width = int(width/scale)
        img = image.resize([width,32],Image.ANTIALIAS)
        img = np.array(img).astype(np.float32)/255.0 - 0.5
        X = img.reshape([1,32,width,1])
        X = X.swapaxes(1,2)
        y_pred = self.model.predict(X)
        y_pred = y_pred[:,:,:]
        out = self.decode(y_pred)
        return out 
def preprocessImage(image):
    width,height = image.size[0],image.size[1]
    scale = height * 1.0 / 32
    width = int(width/scale)
    img = image.resize([width,32],Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)/255.0 - 0.5
    img = np.expand_dims(img,axis=2)
    return img 
def multy_predict(images,n):
    temps = []
    for i in range(n):
        temp = images[i].split(' ')
        image_path = os.path.join('images',temp[0])
        image = Image.open(image_path).convert('L')
        temps.append(preprocessImage(image))
    temps = np.array(temps)
    temps = temps.swapaxes(1,2)
    return temps
def multy_decode(args):
    pred,characters = args
    char_list = []
    pred_text = pred.argmax(axis=1)
    for i in range(len(pred_text)):
        if pred_text[i] != len(characters) -1 and ((not(i>0 and pred_text[i]==pred_text[i-1])) or (i>1 and pred_text[i]==pred_text[i-2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)
if __name__ == '__main__':
    modelpath = 'new_models/weights_5990-01-0.02.h5'
    p1 = Predict(modelpath)

    f = open('data_test.txt','r')
    images = f.readlines()
    """
    for i in range(5):
        temp = images[i].split(' ')
        image_path = os.path.join('images',temp[0])
        image = Image.open(image_path).convert('L')
        label = p1.predict(image)
        print(f'Image {temp[0]} content is {label}')
        plt.imshow(image,cmap='gray')
        plt.show()
    """
    X = multy_predict(images,5)
    result = p1.model.predict(X)
    print(f'result {result.shape}')
    print(len(result))
    #t_result = [result[i] for i in range(len(result))]
    args = []
    for i in range(len(result)):
        args.append((result[i],p1.characters))
    pool = multiprocessing.Pool(processes=5)
    results = pool.map(multy_decode,args)
    print(results)