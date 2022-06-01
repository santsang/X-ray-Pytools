import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.visible_device_list = "0"
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

for script in ["auxNN.py","auxSiamese.py"]:
    execfile(os.path.join(os.path.dirname(os.path.abspath(__file__)),script))
from builtins import input
from google.colab.patches import cv2_imshow
from imutils import paths
import __future__
from IPython.display import clear_output
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer
from tensorflow.keras.callbacks import BaseLogger, Callback, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import argparse, cv2, gc, os, imblearn, json, random, resource, sys, time


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help=">> path to input directory of images")
ap.add_argument("-c", "--checkpoints", required=True,
	help="path to output checkpoint directory")
ap.add_argument("-l", "--learning", type=float, default=1e-3,
	help="Learning rate")
ap.add_argument("-m", "--model", type=str,
	help="path to *specific* model checkpoint to load")
	# eliminate?
ap.add_argument("-p", "--pretrained", type=int, default=None,
	help="*0* for VGG19 and *1* for VGG16")
ap.add_argument("-r", "--run_epoch", type=int, default=25,
	help="epoch to be run to fit the model")
	#recommended epochs
ap.add_argument("-b", "--start_epoch", type=int, default=0,
	help="epoch to restart training at")
ap.add_argument("-s", "--seed", type=int, default=None,
	help=">> enter an *integer*")
args = vars(ap.parse_args())

#############
# Varaibles #
#############
seed=None
while 1:
    oldSeed=seed
    set_all_seed(None)
    seed=2**np.random.choice(16) if not args["seed"] else args["seed"]
    if oldSeed!=seed or args["seed"]:
        rng = set_all_seed(seed)
        break

inputShape = (128, 128, 3)
IMAGE_DIMS=inputShape[0]
BATCH_SIZE = 113
EPOCHS = args["run_epoch"]
INIT_LR = args["learning"]
if args["pretrained"] is not None:
    network=[VGG19, VGG16, EfficientNetB0][args["pretrained"]] #don't add quotes
    fileSyn="Siamese_%s_X-ray"%(["VGG19", "VGG16", "EfficientNetB0"][args["pretrained"]])
else:
    fileSyn="Siamese_X-ray"

plotPath = os.path.join(args["checkpoints"], "%s.pdf"%(fileSyn))
pathPrefix = os.path.splitext(plotPath)[0]
jsonPath = "%s.json"%(os.path.splitext(plotPath)[0])
ckptPath = "%s_best_weights.h5"%(pathPrefix)
modelPath = "%s.h5"%(pathPrefix)
testImages = "%s_test_images.npz"%(pathPrefix)
testLabels = "%s_test_labels.npz"%(pathPrefix)

########
# Main #
########
@memory(proportion=0.8)
def main():
    gc.collect()
    if model:
        del model
    print("Memory is limited to 80%")

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["input"])))
print("No. of file before screening:",len(imagePaths))
rng.shuffle(imagePaths)

data = []
labels = []
for imagePath in imagePaths:
    if re.search("facebook|linkedin|twitter|whatzit",imagePath,re.I) is None:
        label = imagePath.split(os.path.sep)[-2].split("_")
        image = cv2.imread(imagePath)
        if len(image.shape)<3:
            image = cv2.merge(image,image,image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (inputShape[1], inputShape[0]))
        data.append(image)
        labels.append(label)

data=np.array(data, dtype="float") # fit into the model
labels=np.array(labels)
(unique, counts)=np.unique(labels, return_counts=True)
freq = np.asarray((unique, counts)).T
print(">> frequencies of each exclusive class:",freq)
print(">> unique labels before LabelBinarizer():",np.unique(labels, axis=0))
print(">> screened data shape:",data.shape)
print(">> first 5 elements of the first image:",data[1,1,:5])


mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=seed)
del data; gc.collect()

print(">> first 5 trained Y:",trainY[:5])
try:
    print(">> count of trainY:", np.apply_along_axis(np.bincount, 0, mlb.inverse_transform(trainY)))
except: pass

(pairTrain, labelTrain) = make_binary_pairs(trainX, trainY)
(pairTest, labelTest) = make_binary_pairs(testX, testY)

if args["pretrained"] is None:
    featureExtractor=siameseNetwork(inputShape,embeddingDim=len(mlb.classes_))
else:
    featureExtractor=siamesePretrained(inputShape,embeddingDim=len(mlb.classes_)**2,network=network)

model=merge_siamese_model(inputShape,featureExtractor)


print("[INFO] compiling model...")
callbacks = [
	EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=20, verbose=1),
	EpochCheckpoint(args["checkpoints"],fileSyn,rng,every=5,startAt=args["start_epoch"]),
    PlotLearning(),
	ModelCheckpoint(ckptPath, monitor="val_accuracy", mode="max", save_best_only=True,verbose=1)
]
model.compile(loss=contrastive_loss, optimizer=RMSprop(learning_rate=INIT_LR, decay=INIT_LR / (EPOCHS**0.5)), metrics=["accuracy"])


print("[INFO] training model...")
H = model.fit(
	[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=BATCH_SIZE, 
	epochs=EPOCHS,
	callbacks=callbacks,
	verbose=1)
	

y_pred = model.predict([pairTest[:, 0], pairTest[:, 1]])
print(">> pairTest shape:",pairTest.shape)
print(">> y_pred shape:",y_pred.shape)
prob = y_pred[:,0]
print(">> prob:",prob[:10])
labelPred = np.where(prob > 0.5, [1], [0])
print(">> prediction:",labelPred[:10])
print(">> test labels:",labelTest[:10])
#labelTest=[i for a in labelTest for i in a]
print(">> True +ve cases:",np.sum(labelTest))
mapPath="%s_classification.pdf"%(pathPrefix)
confusionMatrix(labelTest,labelPred,["0","1"],fileSyn,mapPath)
reportPath="%s_accuracy.pdf"%(pathPrefix)
accuracyPlot(H,reportPath,title="%s\n(seed: %s)"%(fileSyn,seed))
results = model.evaluate([pairTest[:, 0], pairTest[:, 1]], labelTest)
print(">> test loss & test acc:", results)
K.clear_session()
gc.collect()
del model; del featureExtractor
gc.collect()