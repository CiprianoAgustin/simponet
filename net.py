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
from matplotlib import pyplot as plt

# Directorio padre del proyecto
DIR_PADRE = os.path.abspath("./")

# Importo librería Mask RCNN
sys.path.append(DIR_PADRE)  # Busco librería local
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Ruta a pesos pre-entrenados
PESOS_PATH = os.path.join(DIR_PADRE, "mask_rcnn_coco.h5")

# Directorio de logs y checkpoints de modelo
DIR_LOGS = os.path.join(DIR_PADRE, "logs")

############################################################
#  Configuraciones
############################################################


class NetConfig(Config):
    """Configuración de entrenamiento
    """
    # Nombre de configuración
    NAME = "simponet"

    # Imágenens por GPU
    IMAGES_PER_GPU = 2

    # Número de clases (incluye fondo)
    NUM_CLASSES = 1 + 1  # Fondo + clases

    # Número de pasos por Epoch
    STEPS_PER_EPOCH = 100

    # Evita detecciones con menos de 90% de confianza
    DETECTION_MIN_CONFIDENCE = 0.9

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
        assert subset in ["entrenamiento", "validacion", "prediccion"]
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

        ids_clase = np.zeros([len(info["polygons"])])
        for i, p in enumerate(nombres_clases):
            if p == 'Patient':
                ids_clase[i] = 1
            # Más clases
        ids_clase = ids_clase.astype(int)
        # Retorno máscara y el array de id's de clases de cada instancia
        # Como estoy trabajando con una clase, retorno un array de unos
        return mascara.astype(np.bool), ids_clase

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

    # Condifuración de entrenamiento de prueba
    print("Entrenando cabeceras")
    modelo.train(dataset_train, dataset_val,
                 learning_rate=config.LEARNING_RATE,
                 epochs=60,
                 layers='heads')

############################################################
#  Detección
############################################################

def coloreo(imagen, mascara):
    """Aplica efecto de color
    imagen: imagen RGB
    máscara: máscara de segmentación

    Retorna imagen resultante.
    """
    # Fuerzo un solo canal
    imagen_byn = skimage.color.gray2rgb(skimage.color.rgb2gray(imagen)) * 255
    # Colpaso las máscaras en una sola capa
    mascara = (np.sum(mascara, -1, keepdims=True) >= 1)
    # Copio píxeles de la imagen original
    if mascara.shape[0] > 0:
        imagen_coloreada = np.where(mascara, imagen, imagen_byn).astype(np.uint8)
    else:
        imagen_coloreada = imagen_byn
    return imagen_coloreada

def deteccionYcoloreo(modelo, path_img=None, dir_salida=''):
    assert path_img

    nombres_clases = ['BG', 'Patient']

    # Corro el modelo
    print("Corriendo en {}".format(args.imagen))
    # Leo imagen
    imagen = skimage.io.imread(args.imagen)
    # Detecto objetos
    r = modelo.detect([imagen], verbose=1)[0]
    # Coloreo
    marcacion = coloreo(imagen, r['masks'])
    visualize.display_instances(imagen, r['rois'], r['masks'], r['class_ids'],
                                nombres_clases, r['scores'], making_image=True)
    # Guardo resultado
    arch_salida = "marcacion_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    arch_salida_nom = os.path.join(dir_salida, arch_salida)
    skimage.io.imsave(arch_salida_nom, marcacion)
    print("Saved to ", dir_salida)

def deteccion(modelo, dataset_dir, subset):
    """Corro la detección de imágenes."""
    print("Corriendo en {}".format(dataset_dir))

    os.makedirs('resultados')
    output_dir = os.path.join(os.getcwd(), "resultados/")
    # Leo dataset
    dataset = SimpoDataset()
    dataset.carga_net(dataset_dir, subset)
    dataset.prepare()
    # Cargo imágenes
    output = []
    for image_id in dataset.image_ids:
        # Cargo la imágene y corro la detección
        image = dataset.load_image(image_id)
        # Detecto objetos
        r = modelo.detect([image], verbose=0)[0]
        # Codifico la imagen a RLE
        source_id = dataset.image_info[image_id]["id"]
        rle = masc_a_rle(source_id, r["masks"], r["scores"])
        output.append(rle)
        # Guardo imágen con máscaras
        canvas = visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'], detect=True)
        canvas.print_figure("{}/{}.png".format(output_dir, dataset.image_info[image_id]["id"][:-4]))
    # Guardo a CSV
        output = "ImageId,EncodedPixels\n" + "\n".join(output)
    path_salida = os.path.join(output_dir, "submit.csv")
    with open(path_salida, "w") as f:
        f.write(output)
    print("Guardado a ", output_dir)


############################################################
#  Codificación RLE
############################################################

def rle_codificador(mascara):
    """Codifica una máscara en RLE.
    Retorna un strings de valores separados.
    """
    assert mascara.ndim == 2, "Máscara debe tener dos dimensiones"
    # Aplano a nivel columna
    m = mascara.T.flatten()
    # Computo el gradiente
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # Inidicios de puntos de transición
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convierto el segundo índice en pares
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decodificador(rle, forma):
    """Decodifica una lista RLE y retorna una máscara binaria."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mascara = np.zeros([forma[0] * forma[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mascara.shape[0]
        assert 1 <= e <= mascara.shape[0], "forma: {}  s {}  e {}".format(forma, s, e)
        mascara[s:e] = 1
    # Transposición
    mascara = mascara.reshape([forma[1], forma[0]]).T
    return mascara


def masc_a_rle(id_img, mascara, puntaje):
    "Codifica las máscaras de instancia."
    assert mascara.ndim == 3, "La máscara debe tener tres dimensiones"
    # Si la máscara está vacía retorna la imagen
    if mascara.shape[-1] == 0:
        return "{},".format(id_img)
    # Remuevo superposición de máscaras
    # Multiplico cada instancia por su puntuación
    # Me quedo con el máximo
    orden = np.argsort(puntaje)[::-1] + 1  # descendiente
    mascara = np.max(mascara * np.reshape(orden, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in orden:
        m = np.where(mascara == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_codificador(m)
        lines.append("{}, {}".format(id_img, rle))
    return "\n".join(lines)

############################################################
#  Entrenamiento
############################################################

if __name__ == '__main__':
    import argparse

    # Argumentos
    parser = argparse.ArgumentParser(
        description='Entrenamiento de detección de tumores.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'entrenar', 'detectar' o'inferir'")
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
    parser.add_argument('--subset', required=False,
                        metavar="Subdirectorio del dataset",
                        help="Subset para correr predicciones")
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
    elif args.command == "detectar":
       deteccion(modelo,args.dataset, args.subset)
    elif args.command == "inferir":
       deteccionYcoloreo(modelo, path_img=args.imagen)
    else:
        print("'{}' no se reconoce. "
              "Use 'entrenar', 'detectar' o 'inferir'".format(args.command))
