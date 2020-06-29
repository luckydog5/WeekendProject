import cv2 
import time 
import imutils
import numpy as np 
def style_transfer(weights):
    create = None 
    net = cv2.dnn.readNetFromTorch(weights)
    count = 0
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        current_image = frame.copy()
        current_image = imutils.resize(current_image,224)
        blob = cv2.dnn.blobFromImage(current_image,1.0,(224,224),(103.939, 116.779, 123.680),swapRB=False, crop=False)
        net.setInput(blob)
        start = time.time()
        output = net.forward()
        end_time = time.time()
        #print(f'{1/(end_time-start)} FPS')
        output = output.reshape((3, output.shape[2], output.shape[3]))
        output[0] += 103.939
        output[1] += 116.779
        output[2] += 123.680
        """
        output[0] = output[0].astype(np.uint8)
        output[1] = output[1].astype(np.uint8)
        output[2] = output[2].astype(np.uint8)
        """
        #print(f'before output shape {output.shape}')
        # torch.tanspose(目标坐标顺序)
        #output = output.transpose(1, 2, 0).astype(np.uint8)
        output = output.transpose(1,2,0)
        #output = output.astype(np.uint8)
        #print(f'before output shape {output.shape}')
        """
        if not create:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            create = cv2.VideoWriter('test.mp4',fourcc, 30, (output.shape[1], output.shape[0]), True)
        create.write(output)
        """
        
        cv2.imshow('result',output)
        c = cv2.waitKey(1) & 0xff
        if c == 27:
            break
        
        """
        image_name = 'images/' + str(count) + '.jpg'
        cv2.imwrite(image_name, output, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        count += 1
        """
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    #weights_path = 'models/udnie.t7'
    weights_path = 'models/the_wave.t7'
    style_transfer(weights_path)