import tensorflow as tf
import numpy as np
import os,glob,cv2,sys,argparse,time
models = ['aihole'
        ,'banshankari'
        ,'buddhanagudda'
        ,'hampi'
        ,'lakkundi'
        ,'unkal'
]

def predictor(x_batch):
    modelpredictions = {}
    sess = tf.Session()
    for model in models:
        modelpath = 'Ensemble/'+model
        #print("once")
        saver = tf.train.import_meta_graph(modelpath+'/model_'+model+'.meta')
        saver.restore(sess, tf.train.latest_checkpoint(modelpath+'/'+'./'))
        graph = tf.get_default_graph()
        y_pred = graph.get_tensor_by_name("y_pred:0")
        x= graph.get_tensor_by_name("x:0") 
        y_true = graph.get_tensor_by_name("y_true:0") 
        y_test_images = np.zeros((1, 2)) 
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result=sess.run(y_pred, feed_dict=feed_dict_testing)
        modelpredictions[model] = (result[0])[0]
        #tf.reset_default_graph()
    sess.close()
    #############################################################
    print("\n")
    print("\n")
    for i in modelpredictions:
        print("\n"+i+" : ",modelpredictions[i])

start = time.perf_counter()


# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1] 
filename = dir_path +'/' +image_path
image_size=128
num_channels=3
images = []
# Reading the image using OpenCV
image = cv2.imread(filename)
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)

predictor(x_batch=x_batch)


elapsed = time.perf_counter() - start
print('Elapsed %.3f seconds.' % elapsed)
