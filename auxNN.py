import __future__
from IPython.display import clear_output
from builtins import input
from google.colab.patches import cv2_imshow
from time import sleep
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, ResNet50, VGG16, VGG19
from tensorflow.keras.callbacks import BaseLogger, Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import AveragePooling2D, GaussianDropout, GaussianNoise, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, Activation, Dense, Dropout, Flatten, Input, Conv2D, Lambda, BatchNormalization
from tensorflow.keras.models import load_model, Model, model_from_json, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.util import img_as_ubyte
#from livelossplot.keras import PlotLossesCallback
from imutils import paths, build_montages

import cv2, gc, os, json, random, re, resource, sys, time
import numpy as np
import seaborn as sn
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
#matplotlib.use("Agg")
plt.rcParams["axes.grid.axis"] ="y"
plt.rcParams["axes.grid"] = True
plt.rcParams['axes.facecolor']='white'
plt.rcParams["legend.framealpha"] = 0.1


#########
# Seed #
#########

def set_all_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed) #<>np.random.seed()
    np.random.seed(seed)
    rng=np.random.default_rng(seed)
    tf.compat.v1.set_random_seed(seed)
    CUDA_VISIBLE_DEVICES=""
    return rng



#########
# Model #
#########

"""
transfer learning
"""

def pretrainedAdapted(height, width, depth=3, numClasses=2, finActivation="softmax", network=VGG16):
    """
    adapted from VGG16; use "sigmoid" for penalisation
    """
    pretrained = network(weights='imagenet',
                               include_top=False,
                               input_shape=(height, width, depth))

    pretrained.trainable=False

    x = Flatten()(pretrained.output)
    x = Dense(numClasses,activation=finActivation)(x)
    model = Model(pretrained.input, x)
    print(model.summary())

    return model
    
def pretrained(height, width, depth=3, numClasses=2, finActivation="softmax", network=VGG16):

    pretrained = network(weights='imagenet',
                               include_top=False,
                               input_shape=(height, width, depth))
    print(pretrained.summary())

    pretrained.trainable=False

    layers = [l for l in pretrained.layers]
    inputs = layers[0].output

    x = BatchNormalization()(inputs)
    for i in range(1,(len(layers)-5)):
        x = layers[i](x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = GaussianNoise(0.1)(x)
    x = Dropout(0.5)(x)
    x = Dense(numClasses, activation=finActivation)(x)
    new_model = Model(inputs, x)
    
    new_model=add_regularization(new_model, regularizer=regularizers.l2(2e-4))
    new_model.summary()
    
    return new_model
 

def toyVGG16(height, width, depth=3, classes=2, finActivation="softmax"):
    """
    adapted from VGG16; use "sigmoid" and binary_crossentropy for penalisation
    """
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(height, width, depth))
    vgg_model.summary()
    
    vgg_model.trainable=False

    layers = [l for l in vgg_model.layers]
    x=layers[0].output
    r=0.25
    for i in range(1,11):
        # Need to be touched
        if re.search("conv",layers[i].name,re.I):
            x = layers[i](x)
            x = BatchNormalization(axis=depth)(x)
        elif re.search("pool",layers[i].name,re.I):
            x = layers[i](x)
            x = Dropout(r)(x)
            print(r)
            r+=0.1
        else:
            x = layers[i](x)
        print(x)

    x = AveragePooling2D(pool_size=(4, 4))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = GaussianNoise(0.1)(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation=finActivation)(x)
    new_model = Model(layers[0].input, x)

    new_model.summary()

    return new_model



"""
random weight
"""

def MiniVGGnet(height, width, depth, classes, finActivation="softmax"):

    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format()=="channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    
    # CONV => RELU => POOL
    model.add(Conv2D(32, (3,3), padding = "same", input_shape = inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3))) #consider using average pooling?
    model.add(Dropout(0.25))
    
    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(64, (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(128, (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # classifier
    model.add(Dense(classes))
    model.add(Activation(finActivation))
    
    return model

def MinimalVGGnet(height, width, depth, classes, finActivation="softmax"):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format()=="channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3,3), padding = "same", input_shape = inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.3))
    model.add(AveragePooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation(finActivation))
    return model
    
def avgMiniVGG(height=224, width=224, depth=3, classes=10, finActivation="sigmoid"):
    """
    'softmax' was the recommended activation function
    """
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format()=="channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3,3), padding = "same", input_shape = inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(AveragePooling2D(pool_size=(2,2))) #consider using average pooling?
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation(finActivation))
    return model

def add_regularization(model, regularizer=regularizers.l2(0.0001)):
    import tempfile

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    model_json = model.to_json()

    tempdir = tempfile.mkdtemp() #add by ST
    tmp_weights_path = os.path.join(tempdir, 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    model = model_from_json(model_json)
    
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model

############
# Monitors #
############
class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0, metric = "accuracy"):
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
        self.metric = metric   
    
    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}
        
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                
                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l

        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

class PlotLearning(Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            
    def on_epoch_end(self, epoch, logs={}):
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], ["b--","b-"][i],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], ["r--","r-"][i],
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid(axis="y")

        plt.tight_layout()
        plt.show()

class EpochCheckpoint(Callback):

    def __init__(self, outputPath, fileSyn, rng, every = 5, startAt = 0):
        super(Callback, self).__init__()

        self.outputPath = outputPath
        self.fileSyn = fileSyn if fileSyn else ""
        self.rng = rng
        self.idGauss=[i for i in range(len(model.layers)) if re.search("gauss",model.layers[i].name,re.I)]
        self.every = every
        self.intEpoch = startAt

    def on_epoch_begin(self, epoch, logs=None):
        if self.idGauss:
            for i in self.idGauss:
                model.layers[i].stddev = self.rng.uniform(0,1)
                # rng.random() generates numbers in [0,1)
        self.intEpoch += 1

    def on_epoch_end(self, epochs, logs = {}):
        if (self.intEpoch + 1)%rng.choice(5, 1, replace=True)==0:
            time.sleep(rng.integers(0, 90))
        gc.collect()
        self.intEpoch += 1
        
def accuracyPlot(*args,title):
  """
  e.g., accuracyPlot(H,title="X-ray") (H and fname can be omitted if from H=model.fit(...))
  """
  val=[]
  for k in H.history.keys():
    val.append(k)
  gain= "AUC" if re.search("auc",val[1],re.I) else val[1].title()

  N = len(H.history["loss"])
  title = "" if not title else "%s and Loss on %s"%(gain,title)
  plt.figure(figsize=(15,5))
  plt.suptitle(title)
  for i in range(2):
    plt.subplot(1,2,i+1)
    if i==0:
      plt.plot(H.history[val[1]], "b-", alpha=0.6, label="train_%s"%(val[1]))
      plt.plot(H.history[val[3]], "r-", alpha=0.6, label=val[3])
      plt.ylabel(gain)
      plt.legend(bbox_to_anchor=(-0.01,1),loc="upper left", ncol=2)
    else:
      plt.plot(H.history[val[0]], "b--", alpha=0.6, label="train_%s"%(val[0]))
      plt.plot(H.history[val[2]], "r--", alpha=0.6, label=val[2])
      plt.ylabel("Loss")
      plt.legend(bbox_to_anchor=(1.01,1),loc="upper right", ncol=2)
    plt.xlabel("Epoch #")
    plt.xticks(range(0,N+1),range(1,N+1))
  try:
    plt.savefig(reportPath)
  except:
    print(">> File name is needed to save the graph")
  plt.show()


#################
# Preprocessing #
#################
def add_noise(image):
  """
  Add random Gaussian noise to an image
  """
  var=rng.random()
  imageFloat=isinstance(np.max(image),float)
  if not imageFloat:
    image=np.array(image, dtype="float")/255.0
  gauss =rng.normal(0, var, image.shape) #var=0.1
  noisy=image + gauss
  noisy=np.clip(noisy, 0., 1.)
  if not imageFloat:
    noisy=img_as_ubyte(noisy)
  return noisy

def add_variant_noise(image):
    """
    add different type of noise (may create NaN loss problem)
    """
    var=0.1
    imageFloat=isinstance(np.max(image),float)
    if not imageFloat:
        image=np.array(image, dtype="float")/255.0
    if noiseType is None:
        noisy=image
    elif noiseType == "gauss" or noiseType == "speckle":
        gauss =rng.normal(0, var, image.shape) #var=0.1
        if noiseType == "gauss":
            noisy=image + gauss
        else: noisy=image + image * gauss
    elif noiseType == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [rng.integers(0, i - 1, int(num_salt))
            for i in image.shape]
        out[coords] = 1.
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [rng.integers(0, i - 1, int(num_pepper))
            for i in image.shape]
        out[coords] = 0.
        noisy=out
    elif noiseType == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = rng.poisson(image * vals) / float(vals)

    noisy=np.clip(noisy, 0., 1.)
        
    if not imageFloat:
        noisy=img_as_ubyte(noisy)

    return noisy


##############
# Assessment #
##############
def reportClassication(actual,prediction,classes,reportPath):
  from sklearn.metrics import classification_report
  report=classification_report(actual, prediction,target_names=classes,output_dict=True)
  print(report)
  df = pd.DataFrame(report).transpose()
  pd.set_option('display.precision', 3)
  df.to_csv(reportPath)
  display(df)
  
def confusionMatrix(actual,prediction,classes,titleSyn,cmPath):
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(actual, prediction)
  plt.figure(figsize = (8,6))
  ax=sn.heatmap(cm,annot=True,cmap=plt.cm.Blues,xticklabels=classes,fmt='g')
  ax.set_yticklabels(classes,rotation=360)
  ax.set_xlabel('Actual')
  ax.set_ylabel('Predicted')
  for _, spine in ax.spines.items():
    spine.set_visible(True) #add a frame
  plt.title("Confusion Matrix - %s"%(fileSyn))
  fig = ax.get_figure()
  fig.savefig(mapPath)


################
# Memory Limit #
################
def memory_limit(proportion: float):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * proportion, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemAvailable:':
                free_memory = int(sline[1])
            break
    return free_memory

def memory(proportion=0.8):
    def decorator(function):
        def wrapper(*args, **kwargs):
            memory_limit(proportion)
            try:
                return function(*args, **kwargs)
            except MemoryError:
                mem = get_memory() / 1024 /1024
                print('Remain: %.2f GB' % mem)
                sys.stderr.write('\n\nERROR: Memory Exception\n')
                sys.exit(1)
        return wrapper
    return decorator

        

