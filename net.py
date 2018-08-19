"""
SimpoNet Mask
Entrenamiento sobre Dataset Dicom e implementación de marcación

Escrito por Agustín Cipriano

------------------------------------------------------------

Uso: importar el módulo (ver Jupyter notebooks como ejemplo), o correr desde
       línea de comando:

    # Entrenar un nuevo modelo desde los pesos pre-entrenados
    python net.py entrenar --dataset=ruta --pesos=coco

    # Resumir entrenamiento de modelo entrenado
    python net.py entrenar --dataset=ruta --pesos=ult

    # Aplicar marcacion
    python net.py inferir --pesos=ult --imagen=ruta

"""
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

# Directorio padre del proyecto
DIR_PADRE = os.path.abspath("./")

# Importo librería Mask RCNN
sys.path.append(DIR_PADRE)  # Busco librería local
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Ruta a pesos pre-entrenados
PESOS_PATH = os.path.join(DIR_PADRE, "mask_rcnn_coco.h5")

# Directorio de logs y checkpoints de modelo
DIR_LOGS = os.path.join(DIR_PADRE, "logs")

# Directorio de inferencias
RESULTS_DIR = os.path.join(DIR_PADRE, "inferencias")

############################################################
#  Configuraciones
############################################################


class NetConfig(Config):
    """Configuración de entrenamiento
    """
    # Nombre de configuración
    NAME = "simponet"

    # Imágenens por GPU
    IMAGES_PER_GPU = 1

    # Número de clases (incluye fondo)
    NUM_CLASSES = 1 + 1  # Fondo + clases

    # Número de pasos por Epoch
    STEPS_PER_EPOCH = 100

    # Número de pasos de validación
    VALIDATION_STEPS= 10

    # Manejo de imágenes
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Evita detecciones con menos de 0% de confianza
    DETECTION_MIN_CONFIDENCE = 0

    # Parámetros de entrenamiento
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000
    RPN_NMS_THRESHOLD = 0.9
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    TRAIN_ROIS_PER_IMAGE = 128
    MAX_GT_INSTANCES = 200
    DETECTION_MAX_INSTANCES = 400


class NetInfConfig(NetConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.7

############################################################
#  Dataset
############################################################

class SimpoDataset(utils.Dataset):

    def carga_net(self, dir_dataset, subset):
        """Cargo un subconjunto del dataset.
        dir_dataset: Directorio padre del dataset.
        subset: Subconjunto a cargar (entrenamiento o validación)
        """

        # Añado clases
        self.add_class("Patient", 1, "Patient")

        # Subconjunto de entrenamiento o validación?
        assert subset in ["entrenamiento", "validacion"]
        dir_dataset = os.path.join(dir_dataset, subset)

        # Cargo anotaciones
        # {
        # 'size': xxxxx,
        # 'filename': 'abcdefg',
        # 'file_attr':  {}
        #  "regions": {
        #       'roiname': {
        #           'color': [R,G,B],
        #           'sequence': {
        #               'num': xx,
        #               'type': abcdef,
        #               'num_points': xxx,
        #               'data_x': [...],
        #               'data_y': [...]}},
        #       ... más regiones ...
        #   },
        # }
        anotaciones = json.load(open(os.path.join(dir_dataset, "contornos.json")))
        anotaciones = list(anotaciones.values())  # no se necesitan las keys

        # Evito imágenes no anotadas
        anotaciones = [a for a in anotaciones if a['regions']]

        # Añado las imágenes
        for a in anotaciones:
            # Obtengo las coordenadas x,y del polígono en sequence
            poligonos = [r['sequence'] for r in a['regions'].values()]

            # Obtengo el tamaño de la imagen para generar la máscara
            imagen_path = os.path.join(dir_dataset, a['filename']) + '.jpg'
            imagen = skimage.io.imread(imagen_path)
            altura, ancho = imagen.shape[:2]

            self.add_image(
                "Patient",
                image_id=a['filename'],  # uso filename como ID
                path=imagen_path,
                width=ancho, height=altura,
                polygons=poligonos)


    def load_mask(self, id_img):
        """Genera instancias de máscaras para una imagen
       Returns:
        máscaras: Array booleano de forma (altura, ancho, núm instancias)
        con una máscara por instancia.
        arr_ids_clases: un array unidimensional con los id's de clase de las instancias de máscaras
        """
        # Si no es del dataset de pacientes, lo delego al padre
        info_img = self.image_info[id_img]
        if info_img["source"] != "Patient":
            return super(self.__class__, self).load_mask(id_img)

        # Convierto polígonos a máscara bitmap
        # [alto, ancho, num instancias]
        info = self.image_info[id_img]
        nombres_clases = info["source"]
        mascara = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Obtengo los índices de los pixeles dentro del polígono y los seteo a 1
            rr, cc = skimage.draw.polygon(p['data_y'], p['data_x'])
            mascara[rr, cc, i] = 1

        # ids_clase = np.zeros([len(info["polygons"])])
        # for i, p in enumerate(nombres_clases):
        #     if p == 'Patient':
        #         ids_clase[i] = 1
        #     # Más clases
        # ids_clase = ids_clase.astype(int)
        # Retorno máscara y el array de id's de clases de cada instancia
        # Como estoy trabajando con una clase, retorno un array de unos
        return mascara.astype(np.bool), np.ones([mascara.shape[-1]], dtype=np.int32)
        #return mascara.astype(np.bool), ids_clase

    def image_reference(self, id_img):
        """Retorno el path de la imagen."""
        info = self.image_info[id_img]
        if info["source"] == "Patient":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(id_img)


def entrenar(modelo):
    """Entrenar el modelo."""
    # Dataset de entrenamiento.
    dataset_train = SimpoDataset()
    dataset_train.carga_net(args.dataset, "entrenamiento")
    dataset_train.prepare()

    # Dataset de validación.
    dataset_val = SimpoDataset()
    dataset_val.carga_net(args.dataset, "validacion")
    dataset_val.prepare()

    # Aumentacion de imágenes
    aumentacion = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # Condifuración de entrenamiento de prueba
    print("Entrenando cabeceras")
    modelo.train(dataset_train, dataset_val,
                 learning_rate=config.LEARNING_RATE,
                 epochs=20,
                 augmentation=aumentacion,
                 layers='heads')

    print("Entrenando capas")
    modelo.train(dataset_train, dataset_val,
                 learning_rate=config.LEARNING_RATE,
                 epochs=40,
                 augmentation=aumentacion,
                 layers='all')

def marcacion(imagen, mascara):
    """Aplica la marcacion.
    imagen: imagen
    mascara: mascara de segmentación

    Returns result image.
    """
    # Convierto a escala de grises
    esc_gris = skimage.color.gray2rgb(skimage.color.rgb2gray(imagen)) * 255
    # Copio pixeles
    if mascara.shape[-1] > 0:
        # Colapso las máscaras en una
        mascara = (np.sum(mascara, -1, keepdims=True) >= 1)
        marcado = np.where(mascara, imagen, esc_gris).astype(np.uint8)
    else:
        marcado = esc_gris.astype(np.uint8)
    return marcado

def inferir(modelo, path_img=None):
    assert path_img
    # Correr el modelo
    print("Corriendo en {}".format(args.imagen))
    # Leer imagen
    img = skimage.io.imread(args.imagen)
    # Detectar objetos
    r = modelo.detect([img], verbose=1)[0]
    # Aplico máscara
    img_marcacion = marcacion(img, r['masks'])
    # Guardo resultado
    nombre_arch = "marcacion_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(nombre_arch, img_marcacion)
    print("Guardado en ", nombre_arch)


############################################################
#  Entrenamiento
############################################################

if __name__ == '__main__':
    import argparse

    # Argumentos
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'entrenar' o 'inferir'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/a/dataset/",
                        help='Directorio de dataset')
    parser.add_argument('--pesos', required=True,
                        metavar="/path/a/pesos.h5",
                        help="Path a los pesos (.h5)")
    parser.add_argument('--logs', required=False,
                        default=DIR_LOGS,
                        metavar="/path/a/logs/",
                        help='Directorio de logs y checkpoints (default=logs/)')
    parser.add_argument('--imagen', required=False,
                        metavar="path a imagen",
                        help='Imagen a analizar')
    args = parser.parse_args()

    # Validar argumentos
    if args.command == "entrenar":
        assert args.dataset, "Necesita especificar un dataset"
    elif args.command == "inferir":
        assert args.imagen,\
               "Provea una imagen para inferir"

    print("Pesos: ", args.pesos)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Condiguraciones
    if args.command == "entrenar":
        config = NetConfig()
    else:
        class InferenceConfig(NetConfig):
            # Setea el tamaño del batch a 1
            # Se procesa una imagen por vez. Tamaño del Batch = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Creo modelo
    if args.command == "entrenar":
        modelo = modellib.MaskRCNN(mode="training", config=config,
                                   model_dir=args.logs)
    else:
        modelo = modellib.MaskRCNN(mode="inference", config=config,
                                   model_dir=args.logs)

    # Seleccionar pesos a cargar
    if args.pesos.lower() == "coco":
        pesos = PESOS_PATH
        # Descarga pesos
        if not os.path.exists(pesos):
            utils.download_trained_weights(pesos)
    elif args.pesos.lower() == "ult":
        # Traer últimos pesos
        pesos = modelo.find_last()
    elif args.pesos.lower() == "imagenet":
        # Traer pesos de ImageNet
        pesos = modelo.get_imagenet_weights()
    else:
        pesos = args.pesos

    # Carga pesos
    print("Cargando pesos ", pesos)
    if args.pesos.lower() == "coco":
        # Excluyo las últimas capas porque requieren un número igual de clases
        modelo.load_weights(pesos, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        modelo.load_weights(pesos, by_name=True)

    # Entrenar o inferir
    if args.command == "entrenar":
        entrenar(modelo)
    elif args.command == "inferir":
       inferir(modelo, path_img=args.imagen)
    else:
        print("'{}' no se reconoce. "
              "Use 'entrenar' o 'inferir'".format(args.command))
