
# crnn2.py use Synthetic_Chinese_String Data

Feature

--Based on ResNet50

--Introduce focal_loss

--Remove LSTM layers.

Replacing VGG16 with ResNet50 to gain better feature representations, 

removing LSTM layers to accelerate training process, 

introducing focal-loss can solve class-imbalance to some extent.

Train

python crnn2.py

    --char_set   path to your characters set file, e.g. char_std_5990.txt

    --train_txt  path to your training text file, e.g. data_train.txt

    --test_txt   path to your test text file, e.g. data_test.txt

    --batch_size  128 or 64 or 32

    --image_path  path to your image data folder, e.g. iamges/


data_train.txt contains  training image's name and its corresponding label.

46316437_4182432248.jpg 304 7 263 370 726 2 443 1878 87 295

image name: 46316437_4182432248.jpg

label: 304 7 263 370 726 2 443 1878 87 295

Each number in label stands for a real character in your 

character set file, in char_std_5990.txt it is ['才', '不', '管', '周', '围', '的', '气', '氛', '如', '何']

