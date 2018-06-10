import matplotlib

matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from net.firstnet import NET1
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# ARGUMENTOS Y PARESEO
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path del dataset (directorio de imágenes)")
ap.add_argument("-m", "--model", required=True,
                help="path al modelo resultante")
ap.add_argument("-l", "--labelbin", required=True,
                help="path a tags binarios resultante")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path a gráfico de precisión/pérdida")
args = vars(ap.parse_args())

# EPOCHS, TASA DE APRENDIZAJE, TAMAÑO DE BATCH & DIMENSIONES
EPOCHS = 100
INIT_LEAR_RATE = 1e-3
BATCH_SIZE = 4
IMAGE_DIMS = (512, 512, 1)

# INICIALIZACIÓN
data = []
labels = []

# PATH DE IMÁGENES Y MEZCLA DE IMÁGENES
print("[INFO] cargando imágenes...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# LOOP DE LAS IMÁGENES
for imagePath in imagePaths:
    # CARGA DE IMAGEN, PRE-PROCESAMIENTO Y ALMACENAMIENTO
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    # EXTRACCIÓN DE TAG
    label = imagePath.split(os.path.sep)[-3]
    labels.append(label)

    # ESCALA DE INTENSIDAD DE PIXEL A [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    print("[INFO] data matrix: {:.2f}MB".format(
        data.nbytes / (1024 * 1000.0)))

    # BINARIZACION DE LABEL
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # PARTICION DE DATA ENTRE ENTRENAMIENTO Y TESTING
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.2, random_state=42)

    # GENERADOR DE IMÁGENES
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    # MODELO
    print("[INFO] inicializando modelo...")
    model = NET1.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                       depth=IMAGE_DIMS[2], classes=len(lb.classes_))
    opt = Adam(lr=INIT_LEAR_RATE, decay=INIT_LEAR_RATE / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # ENTRENAMIENTO
    print("[INFO] entrenamiento de red...")
    H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BATCH_SIZE,
        epochs=EPOCHS, verbose=1)

    # ALMACENAMIENTO DEL MODELO
    print("[INFO] serializing network...")
    model.save(args["model"])

    # ALMACENAMIENTO DE TAGS BDEFC
    print("[INFO] serializing label binarizer...")
    f = open(args["labelbin"], "wb")
    f.write(pickle.dumps(lb))
    f.close()

    # GRÁFICO DE PRECISIÓN Y PÉRDIDA
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(args["plot"])
