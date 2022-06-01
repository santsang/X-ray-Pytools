import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.visible_device_list = "0"
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
tf.compat.v1.disable_eager_execution()

for pkg in ["auxNN.py", "auxSiamese.py"]:
    execfile(os.path.join(os.path.dirname(os.path.abspath(__file__)),pkg))
import __future__
from IPython.display import clear_output
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import BaseLogger, Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
#from livelossplot.keras import PlotLossesCallback
import tensorflow.keras.backend as K
import numpy as np
import seaborn as sn
import argparse, cv2, os, json, random, resource, sys, time
import matplotlib.pyplot as plt

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
	help="*0* for efficientNet, *1* ResNet50, *2* VGG16,...")
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
    
IMAGE_DIMS=128
BS = 8
INIT_LR = args["learning"]
EPOCHS = args["run_epoch"]
if args["pretrained"] is None:
    fileSyn="miniNet_X-ray"
else:
    network=[EfficientNetB0, ResNet50, VGG16, VGG19][args["pretrained"]] #don't add quotes
    fileSyn="%s_X-ray"%(["EfficientNet", "ResNet50", "VGG16", "VGG19"][args["pretrained"]])

print("[INFO] compiling model...")
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
    print("Memory is limited to 80%")
        

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["input"]))
print("No. of file before screening:",len(imagePaths))
rng.shuffle(imagePaths)

data = []
labels = []
	
for imagePath in imagePaths:
    if re.search("facebook|linkedin|twitter|whatzit",imagePath,re.I) is None:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2].split("_")[0]
        image = cv2.imread(imagePath)
        if len(image.shape)<3:
            image = cv2.merge([image]*3)
        else: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_DIMS, IMAGE_DIMS))
        data.append(image)
        labels.append(label)
        
data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

print(">> data shape",data.shape)
print(">> unique labels:",set(labels))
print(">> the first 5 elements of the first image",data[1,1,:5],"...")
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(">> list of classes:",list(lb.classes_))
print(">> shape of images:",data.shape)


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=seed)
del data
gc.collect()
print(">> Training images before SMOTE:",trainX.shape)
print(">> Reponses/Labels:",trainY.shape)

sm = SMOTE(random_state=seed)
train_rows=len(trainX)
print("training image before SMOTE",train_rows)
X_train = trainX.reshape(train_rows,-1)
    
X_train, y_train = sm.fit_resample(X_train, trainY)
X_train = X_train.reshape(-1,IMAGE_DIMS,IMAGE_DIMS,trainX.shape[3])
print(">> Training images after SMOTE:",X_train.shape)

aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=[0.2,1],
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest",
	brightness_range=[0.2,1.2],
	preprocessing_function=add_noise
	) 	

if args["model"] is None:
    if not args["pretrained"]:
        model = MinimalVGGnet(width=IMAGE_DIMS, height=IMAGE_DIMS, depth=3, classes=len(lb.classes_))
    else:
        model = pretrained(IMAGE_DIMS, IMAGE_DIMS, depth=3, numClasses=len(lb.classes_),finActivation="softmax",network=network)

    print("[INFO] compiling model...")
    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="kl_divergence", optimizer=opt, metrics=["accuracy"])

else:
	print("[INFO] loading {}...".format(args["model"]))
	try: model = load_model(args["model"])
	except: model = load_model(ckptPath)
	# update the learning rate
	print("[INFO] old learning rate: {}".format(
		K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, args["learning"])
	print("[INFO] new learning rate: {}".format(
		K.get_value(model.optimizer.lr)))

print("[INFO] compiling model...")
callbacks = [
	EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=20, verbose=1),
	EpochCheckpoint(args["checkpoints"],fileSyn,model,rng,every=5,startAt=args["start_epoch"]),
	TrainingMonitor(plotPath,jsonPath=jsonPath,startAt=args["start_epoch"]),
	ModelCheckpoint(ckptPath, monitor="val_accuracy", mode="max", save_best_only=True,verbose=1)
]

if args["seed"]:
    os.remove(jsonPath)

H=model.fit(
	x=aug.flow(X_train, y_train, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=args["run_epoch"],
	callbacks=callbacks,
	verbose=1)

reportPath="%s_accuracy.pdf"%(pathPrefix)
accuracyPlot(H,reportPath,title="%s\n(seed: %s)"%(fileSyn,seed))
	
print("[INFO] evaluating network...")
preds = model.predict(testX, batch_size=BS)
print(">>preds:",preds[:10,])
K.clear_session()
gc.collect()
del model
Y_pred_classes = preds.argmax(axis=1)
Y_true = testY.argmax(axis=1)
print(">>predicted classes:",Y_pred_classes[:10,])
print(">>actual:",Y_true[:10,])
reportPath="%s_classication.csv"%(os.path.splitext(plotPath)[0])
reportClassication(Y_true,Y_pred_classes,lb.classes_,reportPath)
mapPath="%s_heatmap.pdf"%(os.path.splitext(plotPath)[0])
confusionMatrix(Y_true,Y_pred_classes,lb.classes_,fileSyn,mapPath)
gc.collect()

