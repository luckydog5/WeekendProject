import os
import argparse	
import json
import threading
import numpy as np
from PIL import Image
import traceback
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.optimizers import Adadelta
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.layers.wrappers import TimeDistributed
#from parameter import *
#K.set_learning_phase(0)


GPU_ID_LIST = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID_LIST
alpha = 0.25
gama = 0.5
img_h = 32
img_w = 280
#batch_size = 128
#maxlabellength = 32
maxlabellength = 10
GPU_NUM = 1
lstm_unit_num = 256
#batch_size = 128 * GPU_NUM

train_size = 3607567
test_size = 36440

#encode_dct =  {}


def parse_arguments():

	parser = argparse.ArgumentParser(description='Some parameters.')
	parser.add_argument(
		"--char_set",
		type=str,
		help="The path to the char_set, default char_set is ./chn.txt",
		default="char_map.txt"
	)
	parser.add_argument(
		"--model_path",
		type=str,
		help="Path to load  pre-model.",
		default="resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
	)
	parser.add_argument(
		"--train_txt",
		type=str,
		help="Where to load the train.txt, default path is train.txt.",
		default="train.txt"
	)
	parser.add_argument(
		"--test_txt",
		type=str,
		help="Where to load the test.txt, test.txt.",
		default="test.txt"
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		help="Batch size, default is 128.",
		default=128

	)
	parser.add_argument(
		"--image_path",
		type=str,
		help="Image path.",
		default="/data/denghailong/OCR_textrender/"
	)
	return parser.parse_args()
def identity_block(input_tensor, kernel_size, filters, stage, block):
	"""The identity block is the block that has no conv layer at shortcut.

	# Arguments
		input_tensor: input tensor
		kernel_size: defualt 3, the kernel size of middle conv layer at main path
		filters: list of integers, the filterss of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names

	# Returns
		Output tensor for the block.
	"""
	filters1, filters2,filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1

	# conv and bn layer's name 
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1,1),kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size,kernel_initializer='he_normal', padding='same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1,1),kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = layers.add([x, input_tensor])
	x = Activation('relu')(x)

	return x
def conv_block(input_tensor, kernel_size, filters, stage,block,strides=(2,2)):
	"""conv_block is the block that has a conv layer at shortcut

	# Arguments
		input_tensor: input tensor
		kernel_size: defualt 3, the kernel size of middle conv layer at main path
		filters: list of integers, the filterss of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names

	# Returns
		Output tensor for the block.

	Note that from stage 3, the first conv layer at main path is with strides=(2,2)
	And the shortcut should have strides=(2,2) as well
	"""
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3

	else:

		bn_axis = 1


	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1,1), kernel_initializer='he_normal',strides= strides, name=conv_name_base + '2a')(input_tensor)

	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size, kernel_initializer='he_normal',padding = 'same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1,1), kernel_initializer='he_normal', name = conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base +'2c')(x)

	shortcut = Conv2D(filters3,(1,1),kernel_initializer='he_normal',strides=strides,name=conv_name_base + '1')(input_tensor)

	shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = layers.add([x, shortcut])
	x = Activation('relu')(x)

	return x



def get_session(gpu_fraction=0.95):

	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

	if num_threads:
		return tf.Session(config=tf.ConfigProto(
			gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False))

def readfile(filename):
	# construct a dictionary, image_name : label.
	res = []
	with open(filename, 'r') as f:
		lines = f.readlines()
		for i in lines:
			res.append(i.strip())
	dic = {}
	"""
	for i in res:
		img_name, label = i.split('\t')
		if len(img_name)==0 or len(label)==0 or len(label)>30:
			continue
		dic[img_name] = label
	"""
	# Synthetic_Chinese_String_Dataset
	# format of training data : image_name (int)label splited by space.
	for i in res:
		p = i.split(' ')
		dic[p[0]] = p[1:]		
	return dic

class random_uniform_num():
	
	def __init__(self, total):
		self.total = total
		self.range = [i for i in range(total)]
		np.random.shuffle(self.range)
		self.index = 0
	def get(self, batchsize):
		r_n=[]
		if(self.index + batchsize > self.total):
			r_n_1 = self.range[self.index:self.total]
			np.random.shuffle(self.range)
			self.index = (self.index + batchsize) - self.total
			r_n_2 = self.range[0:self.index]
			r_n.extend(r_n_1)
			r_n.extend(r_n_2)
		else:
			r_n = self.range[self.index : self.index + batchsize]
			self.index = self.index + batchsize

		return r_n

cur_line = None



def gen(data_file, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
	# read traing.txt
	# return dic={'image_name':label}
	image_label = readfile(data_file)
	_imagefile = [i for i, j in image_label.items()]
	#x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
	x = np.zeros((batchsize, imagesize[1], imagesize[0], 1), dtype=np.float)
	labels = np.ones([batchsize, maxlabellength]) * 10000
	input_length = np.zeros([batchsize, 1])
	label_length = np.zeros([batchsize, 1])

	r_n = random_uniform_num(len(_imagefile))
	_imagefile = np.array(_imagefile)
	idx = 0
	while 1:
		for i in range(0, len(r_n.range)):
			fname = _imagefile[i]
			
			img_f = os.path.join(image_path, fname).strip()
			img1 = Image.open(img_f).convert('L')
			img = np.array(img1, 'f') / 255.0 - 0.5
			#转成w * h
			x[idx] = np.expand_dims(img, axis=2).swapaxes(0,1)
			#print(x.shape)
			#x = x.swapaxes(1,2)
			#print(x.shape)
			label = image_label[fname]
			#label_idx_list = [encode_dct[c] for c in label]
			#print (str, len(str))
			label_length[idx] = len(label)
			
			#不太明白这里为什么要减去2
			#跟两个MaxPooling有关系?
			input_length[idx] = imagesize[1] // 8 - 2
			#labels[idx, :len(str)] = [int(k) - 1 for k in str]
			labels[idx, :len(label)] = [int(i) -1 for i in label]
			if len(labels[idx]) > maxlabellength:
				print ("LEN DSHJ : ", len(labels[idx]))
			#print (x[idx].shape, input_length[idx], labels[idx], label_length[idx])
			idx += 1
			if idx == batchsize:
				idx = 0
				#print ("Watch : ", img_f , str)
				#print([int(k) - 1 for k in str])
				inputs = {'the_input': x,
					'the_labels': labels,
					'input_length': input_length,
					'label_length': label_length,
					}
				outputs = {'ctc': np.zeros([batchsize])}
				#print (new_input_length, new_label_length, new_labels.shape, new_labels)
				yield (inputs, outputs)

# # Loss and train functions, network architecture
def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	# if use lstm:
	# 	the 2 is critical here since the first couple outputs of the RNN
	# 	tend to be garbage.
	# else if not use lstm:
	#	there is no need to skip first two outputs
	#y_pred = y_pred[:, 2:, :]

	ctc_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
	p = K.exp(-ctc_loss)
	focal_loss = alpha*K.pow((1-p), gama)*ctc_loss
	return focal_loss


def get_model(training, img_h, nclass):

	if K.image_data_format() == 'channels_last':
		bn_axis = 3

	else:

		bn_axis = 1
	input_shape = (None, img_h, 1)	 # (128, 64, 1)
	# Bulid Networkw
	inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

	"""
	# Convolution layer (VGG)
	inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)  # (None, 128, 64, 64)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

	inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

	inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)
	# use dropout may result in unconvergence.
	# inner = Dropout(0.1)(inner)


	inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 512)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

	inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 4, 512)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	"""
	#x = ZeroPadding2D((3,3))(inputs)
	x = Conv2D(64,(3,3), padding='same', strides=(1,1),name='conv1')(inputs)
	x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((1,2),strides=(1,2))(x)
	# w = w/4, h = h/4
	x = conv_block(x,3,[64,64,256],stage=2,block='a',strides=(1,1))
	x = identity_block(x,3,[64,64,256], stage=2, block='b')
	x = identity_block(x,3,[64,64,256],stage=2, block='c')
	# no change
	x = conv_block(x,3,[128,128,512], stage=3, block='a')
	x = identity_block(x, 3, [128,128,512], stage=3, block='b')
	x = identity_block(x, 3, [128,128,512], stage=3, block='c')
	x = identity_block(x, 3, [128,128,512], stage=3, block='d')
	# w = w/8, h = h/8
	x = conv_block(x, 3, [256,256,1024], stage=4,block='a')
	x = identity_block(x, 3, [256,256,1024], stage=4, block='b')
	x = identity_block(x, 3, [256,256,1024], stage=4, block='c')
	x = identity_block(x, 3, [256,256,1024], stage=4, block='d')
	x = identity_block(x, 3, [256,256,1024], stage=4, block='e')
	x = identity_block(x, 3, [256,256,1024], stage=4, block='f')
	# w = w/16, h = h/16
	x = conv_block(x, 3, [512,512,2048], stage=5, block='a')
	x = identity_block(x, 3, [512,512,2048], stage=5, block='b')
	x = identity_block(x, 3, [512,512,2048], stage=5, block='c')
	x = AveragePooling2D((1,2),name='avg_final')(x)
	# w = w/32, h = h/32  ---->(7,7,2048)
	#x = AveragePooling2D((7,7), name='avg_pool')(x)
	# output is (1,1,2048)
	#x = Conv2D(2048, (2,2), padding='same', kernel_initializer='he_normal', name='conv_final')(x)
	# CNN to RNN
	#inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
	inner = TimeDistributed(Flatten(), name='flatten')(x)
	"""
	lstm_1 = LSTM(lstm_unit_num,return_sequences=True,kernel_initializer='he_normal',name='lstm1')(inner)
	lstm_1b = LSTM(lstm_unit_num,return_sequences=True,go_backwards=True,kernel_initializer='he_normal',name='lstm1_b')(inner)
	lstm1_merged = add([lstm_1,lstm_1b])
	lstm1_merged = BatchNormalization()(lstm1_merged)

	lstm_2 = LSTM(lstm_unit_num,return_sequences=True,kernel_initializer='he_normal',name='lstm2')(lstm1_merged)
	lstm2_b = LSTM(lstm_unit_num,return_sequences=True,go_backwards=True,kernel_initializer='he_normal',name='lstm2_b')(lstm1_merged)
	lstm2_merged = concatenate([lstm_2,lstm2_b])
	lstm2_merged = BatchNormalization()(lstm2_merged)
	
	inner = Dense(nclass, kernel_initializer='he_normal',name='dense2')(lstm2_merged) #(None, 32, 63)
	"""
	inner = Dense(nclass,kernel_initializer='he_normal',name='dense2')(inner)
	y_pred = Activation('softmax', name='softmax')(inner)

	labels = Input(name='the_labels', shape=[None], dtype='float32') # (None ,8)
	input_length = Input(name='input_length', shape=[1], dtype='int64')	 # (None, 1)
	label_length = Input(name='label_length', shape=[1], dtype='int64')	 # (None, 1)

	# Keras doesn't currently support loss funcs with extra parameters
	# so CTC loss is implemented in a lambda layer
	# use focal-ctc loss. alpha = 0.25, gama = 0.5

	#loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)
	
	focal_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
	

	#Then compile model with focal-loss
	model = None
	if training:
		model =  Model(inputs=[inputs, labels, input_length, label_length], outputs=focal_loss)
	#model =  Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
	else:
		model = Model(inputs=inputs, outputs=y_pred)
		return model
	model.summary()
	#multi_model = multi_gpu_model(model, gpus=GPU_NUM)
	save_model = model
	ada = Adadelta()
	#multi_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
	#multi_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada, metrics=['accuracy'])
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
	return model


if __name__ == '__main__':
	args = parse_arguments()
	char_set = open(args.char_set, 'r', encoding='utf-8').readlines()
	
	char_set = ''.join([ch.strip('\n') for ch in char_set[1:]] + ['卍'] )
	# char_set = ''.join([chr(i) for i in range(32, 127)] + ['卍'])
	#. There always arise bugs, len(classes) > len(labels) + 1
	nclass = len(char_set) 
	#K.set_session(get_session())
	# reload(densenet)
	model = get_model(True, img_h, nclass)
	modelPath = args.model_path
	"""
	if os.path.exists(modelPath):
		print("Loading model weights...")
		model.load_weights(modelPath,by_name=True)
		print('done!')
	"""
	train_loader = gen(args.train_txt, args.image_path, batchsize=args.batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
	test_loader = gen(args.test_txt, args.image_path, batchsize=args.batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
	#train_loader = gen('./output/default/tmp_labels.txt', './output/default/', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
	#test_loader = gen('./test/default/tmp_labels.txt', './test/default/', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
	checkpoint = ModelCheckpoint(filepath='./new_models/weights_5990-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=False, save_weights_only=True)
	checkpoint.set_model(model)
	#lr_schedule = lambda epoch: 0.0005 * 0.4**epoch
	#lr_schedule = lambda epoch: 0.005 * 20 * 0.4 / (epoch + 1)
	#lr_schedule = lambda epoch: 0.00135 * 2 * 0.33**epoch
	lr_schedule = lambda epoch: 0.0005 * 1 * 0.55**epoch
	
	learning_rate = np.array([lr_schedule(i) for i in range(30)])
	changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
	earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
	tensorboard = TensorBoard(log_dir='./new_models/logs', write_graph=True)
	print('-----------Start training-----------')
	model.fit_generator(train_loader,
		steps_per_epoch = train_size //args.batch_size,
		epochs = 30,
		initial_epoch = 0,
		validation_data = test_loader,
		validation_steps = test_size //args.batch_size,
		callbacks = [checkpoint, earlystop, changelr, tensorboard])
		#callbacks = [checkpoint, changelr, tensorboard])
		#callbacks = [checkpoint, tensorboard])




