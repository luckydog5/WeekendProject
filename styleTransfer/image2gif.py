import os 
from PIL import Image,ImageDraw



def img2gif(save_dir):
    frames = []
    image_names = os.listdir(save_dir)
    for image in sorted(image_names,key=lambda name: int(''.join(i for i in name if i.isdigit()))):
        frames.append(Image.open(save_dir+'/'+image))
    frames[0].save('styleTransfer.gif',format='GIF',append_images=frames[1:],save_all=True,duration=80,loop=0)


if __name__ == '__main__':

    save_dir = 'images/'
    img2gif(save_dir)