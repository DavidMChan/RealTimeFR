import tflearn
import numpy as np

def get_emotion(ival):
	emotion = np.argmax(ival)
	switcher = {
                0: ("Neutral",(255,255,255)),
                1: ("Happiness",(0,255,255)),
                2: ("Sadness",(255,144,30)),
                3: ("Surprise",(255,255,0)),
                4: ("Fear",(255,0,255)),
                5: ("Disgust",(0,255,0)),
		6: ("Anger",(0,0,255)),
                7: ("Contempt",(0,153,255))
	}
	return switcher.get(emotion, ("N/A",(0,0,0)))

def get_model(model_name):
    # First we load the network
    print("Setting up neural networks...")
    n = 18

    # Real-time data preprocessing
    print("Doing preprocessing...")
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(per_channel=True, mean=[0.573364,0.44924123,0.39455055])

    # Real-time data augmentation
    print("Building augmentation...")
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_crop([32, 32], padding=4)

    #Build the model (for 32 x 32)
    print("Shaping input data...")
    net = tflearn.input_data(shape=[None, 32, 32, 3],
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug)
    net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)

    print("Carving Resnext blocks...")
    net = tflearn.resnext_block(net, n, 16, 32)
    net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
    net = tflearn.resnext_block(net, n-1, 32, 32)
    net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
    net = tflearn.resnext_block(net, n-1, 64, 32)

    print("Erroding Gradient...")
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    net = tflearn.fully_connected(net, 8, activation='softmax')
    opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    net = tflearn.regression(net, optimizer=opt,
                             loss='categorical_crossentropy')
                                                     
    print("Structuring model...")
    model = tflearn.DNN(net, tensorboard_verbose=0,
                        clip_gradients=0.)

    # Load the model from checkpoint
    print("Loading the model...")
    model.load(model_name)

    return model
