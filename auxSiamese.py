from builtins import input
from google.colab.patches import cv2_imshow
import __future__
from tensorflow.keras.layers import AveragePooling2D, GaussianDropout, GaussianNoise, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, Activation, Dense, dot, Dropout, Flatten, Input, Conv2D, Lambda, BatchNormalization
from tensorflow.keras.models import load_model, Model, model_from_json, Sequential
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths, build_montages
import cv2
tf.compat.v1.disable_eager_execution()


#############
# auxillary #
#############  

def indices(anyList, criterion):
  """
  find indices of the same value in a list
  """
  return [i for i, x in enumerate(anyList) if x==criterion]


#################
# preprocessing #
#################    

def make_pairs(images, labels):
    """
    pairing images and labels (one pair for training dataset and one for test dataset)
    """
    pairImages = []
    pairLabels = []
    # calculate the total number of classes in the dataset
    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
    # build a list of indexes for each class label that provides the indexes for all examples with a given label

    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        currentImage = images[idxA]
        label = labels[idxA]
        idxB = rng.choice(idx[label])
        posImage = images[idxB]
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        
        negIdx = np.where(labels != label)[0]
        negImage = images[rng.choice(negIdx)]
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    
    return (np.array(pairImages), np.array(pairLabels))

def make_binary_pairs(images, labels):
    """
    pairing images and labels (one pair for training dataset and one for test dataset)
    """
    limit=10
    pairImages = []
    pairLabels = []
    # calculate the total number of classes in the dataset
    uniqueLabels=np.unique(labels, axis=0)
    numClasses=len(uniqueLabels)
    print("(%s classes:)"%(numClasses),uniqueLabels)
    idC=[j for i in range(len(labels)) for j in range(numClasses) if all(uniqueLabels[j]==labels[i])]
    # class of each label
    run=0
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        currentImage = images[idxA]
        member=idC[idxA]
        if run<limit:
            print(">> Unique Labels:",uniqueLabels[member])
            print(">> Current Labels:",labels[idxA])
            print(">> Same class? (expect 'True'):",all(uniqueLabels[member]==labels[idxA]))
            pass
        members=[i for i, x in enumerate(idC) if x==member]
        idxB = rng.choice(members)
        if run<limit:
            print(">> idxA and idxB in the same club (expect 'True')?: ",idC[idxB]==member)
        posImage = images[idxB]
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        
        negIdx = [i for i, x in enumerate(idC) if x!=member]
        notIdxB = rng.choice(negIdx)
        negImage = images[notIdxB]
        if run<limit:
            print(">> idxA and notIdxB in the same club (expect 'False')?: ",notIdxB in members)
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
        run+=1 
    
    return (np.array(pairImages), np.array(pairLabels))



def toyData(inputImages,responses,cond=None,fontSize=0.6,fontThickness=3,idS=0):
  """
  Designed based on the mnist dataset and assumed the condition is 2-level
  """
  (x,y)=(128,196)
  images = []
  labels = []
  colDict = {"B": (255,0,0), "G": (0,255,0), "Y": (102,255,255)}
  for i in range(len(responses)):
    img=inputImages[i]
    img=cv2.resize(img,(x,y))
    if len(img.shape)<3:
      img=cv2.label=cv2.merge([img]*3)
    label=responses[i]
    if cond is None:
        col=rng.choice(["B","G","Y"]) #truly random
        label=(label,col)
        print(label)
    else:
        col="Y" if re.search(cond,label[idS],re.I) else "B"
    colour=colDict[col]
    text=str(label[0])
    text=text if len(text)<3 else text[:3]
    cv2.putText(img, "%s"%(text), (3,y-12), cv2.FONT_HERSHEY_SIMPLEX, fontSize,
		colour, fontThickness)

    images.append(img)
    labels.append(label)
  return (images,labels)



####################
# Visualisation/QC #
####################    
def view_montage(pairedImages,pairLabels,nbr=30):
    """
    allow see an array of diagrams at one-go
    """
    nbr=nbr if nbr%6==0 else 6*(nbr//6)
    print("size before np.stack():",pairedImages[0][0].shape)
    images = []
    for i in random.choices(np.arange(0, len(pairedImages)), k=nbr):
        imageA = pairedImages[i][0]
        imageB = pairedImages[i][1]
        label = pairLabels[i]
        (x,y)=(128, 196)
        imageA = cv2.resize(imageA, (x,y), interpolation=cv2.INTER_LINEAR)
        imageB = cv2.resize(imageB, (x,y), interpolation=cv2.INTER_LINEAR)
        if len(imageA.shape)<3:
            imageA=cv2.merge([imageA] * 3)
        if len(imageB.shape)<3:
            imageB=cv2.merge([imageB] * 3)
        pair = np.hstack([imageA, imageB])
        (y1,x1)=pair.shape[:2]
        output = np.zeros((y1+4, x1+4, 3), dtype="uint8") #(36, 60, 3)
        output[2:y1+2,2:x1+2,:] = pair # [4:32,0:56,:]
        (text, colour) = ("-ve pair",(0, 0, 255)) if label[0] == 0 else ("+ve pair",(0, 255, 0))
        vis = cv2.resize(output, (256, 196), interpolation=cv2.INTER_LINEAR)
        cv2.putText(output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
		colour, 3)
        images.append(output)

    print(">> Inspecting pairs")
    montage = build_montages(images, (256, 196), (6, nbr//6))[0]
    cv2_imshow(montage)
    cv2.waitKey(0)

#########
# model #
#########
def siameseNetwork(inputShape,embeddingDim=10):
    
    input = Input(inputShape)
  
    x = Conv2D(64, (2, 2), activation="relu", padding = "same")(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (2, 2), activation="relu", padding = "same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (2, 2), activation="relu", padding = "same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (2, 2), activation="relu", padding = "same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = AveragePooling2D(pool_size=(4,4))(x)
    x = Flatten()(x)
    x = Dense(embeddingDim)(x)
    x = tf.nn.l2_normalize(x, axis=-1)
    embedding_network = Model(input, x)
    
    embedding_network=add_regularization(embedding_network, regularizer=regularizers.l2(2e-5))
    
    embedding_network.summary()
    
    return(embedding_network)


def siamesePretrained(inputShape,embeddingDim=12,network=VGG19,TRAINABLE=False):
    
    netInputs=Input(inputShape)

    pretrained = network(weights='imagenet',include_top=False,input_shape=inputShape)

    pretrained.trainable=TRAINABLE
      #freeze the layers

    layers = [l for l in pretrained.layers]


    x = layers[1](netInputs)
    for i in range(2,len(layers)):
        if not re.search("block5",layers[i].name,re.I):
            x = layers[i](x)
            print(x)
        else:
            print(">>Not included layer and thereafter:", layers[i].name)
            break

    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(embeddingDim)(x)

    embedding_network=Model(netInputs, x)

    embedding_network=add_regularization(embedding_network, regularizer=regularizers.l2(2e-4))

    embedding_network.summary()
    
    return embedding_network


def merge_siamese_model(inputShape,featureExtractor):
    print("[INFO] building siamese network...")
    imgA = Input(inputShape)
    imgB = Input(inputShape)
    featsA = featureExtractor(imgA) #first arm
    featsB = featureExtractor(imgB) #second arm
    distance = Lambda(euclidean_distance)([featsA, featsB])
    outputs = Dense(1, activation="sigmoid",name="predict")(distance)
    model = Model(inputs=[imgA, imgB], outputs=outputs)

    model.summary()
    
    return model
    
##############
# evaluation #
##############
def contrastive_loss(y, preds, margin=1):
	y = tf.cast(y, preds.dtype)
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	return loss
	
def euclidean_distance(vectors):
	(featsA, featsB) = vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))


