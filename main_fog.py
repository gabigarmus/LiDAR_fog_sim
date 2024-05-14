#Commum libraries
import os
import time
import sys
import copy
import socket
import pandas
import logging
import argparse
#import pcl
import tempfile

import numpy as np 
import pickle as pkl
import matplotlib as mpl
import matplotlib.cm as cm
import pyqtgraph.opengl as gl


from glob import glob
from typing import List
from pathlib import Path
from plyfile import PlyData

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtGui 
from PyQt5.QtWidgets import QApplication, QLabel

#====================================================================================================================
#Import library fog

import multiprocessing as mp

from fog_simulation import ParameterSet, RNG, simulate_fog

from SeeingThroughFog.tools.DatasetViewer.dataset_viewer import load_calib_data, read_label
from SeeingThroughFog.tools.DatasetFoggification.beta_modification import BetaRadomization
from SeeingThroughFog.tools.DatasetFoggification.lidar_foggification import haze_point_cloud

#-------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', type=str, help='path to where you store your datasets',
                    default=str(Path.home() / 'datasets'))
parser.add_argument('-e', '--experiments', type=str, help='path to where you store your OpenPCDet experiments',
                    default=str(Path.home() / 'repositories/PCDet/output'))
args = parser.parse_args()
DATASETS_ROOT = Path(args.datasets)
FOG =  DATASETS_ROOT / 'DENSE/SeeingThroughFog/lidar_hdl64_strongest_fog_extraction'


def get_extracted_fog_file_list(dirname: str) -> List[str]:

        file_list = [y for x in os.walk(dirname) for y in glob(os.path.join(x[0], f'*.bin'))]

        return sorted(file_list)

app = QApplication(sys.argv)
#Aqui foi feito um codigo para carregar os arquivos .bin



def read_kitti_bin(bin_path):
    """Função para ler o arquivo .bin do dataset KITTI"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def save_kitti_bin(points, bin_path):
    """Função para salvar a nuvem de pontos em formato .bin"""
    points.tofile(bin_path)
    print(f"Nuvem de pontos salva em {bin_path}")

class CloudLoader:
    def __init__(self) -> None:
        self.dataset = None
        self.workspace_folder = None
        self.min_value = -1
        self.max_value = -1
        self.num_features = 4
        self.extension = 'bin'
        self.d_type = np.float32
        self.intensity_multiplier = 255
        self.color_dict = {2: 'some_color', 6: 'not available'}
        self.success = False
        self.color_name = None 
        self.color_label = None
        self.color_feature = 2
        self.extracted_fog_file_list = []  # Inicializando como uma lista vazia
        self.point_size = 3
        self.threshold = 50
        self.color_label = QLabel()
        self.extracted_fog_index = -1
        self.file_name = None
        self.noise = 10
        self.noise_min = 0
        self.noise_max = 20
        self.current_pc = None
        self.gain = True
        self.noise_variant = 'v4'
        self.color_name = self.color_dict[self.color_feature]
        self.file_list = []  # Inicializar a lista de arquivos

        self.p = ParameterSet(gamma=0.000001,
                              gamma_min=0.0000001,
                              gamma_max=0.00001,
                              gamma_scale=10000000)

        self.p.beta_0 = self.p.gamma / np.pi
        self.row_height = 20
 
#-----------------------------------------------------------------------------------
#load data
    #def set_workspace_folder(self, folder_path: str) -> None:
        #self.workspace_folder = os.path.join(folder_path, "data")

    def load_clouds_from_workspace(self, filename: str) -> None:
        self.file_list = [filename]
        print(f"Point cloud loaded from file: {filename}")


        #self.file_list = [filename]
        #print(f"Point cloud loaded from file: {filename}")


#-----------------------------------------------------------------------------------
 
    def load_pointcloud(self, filename: str) -> np.ndarray:
        try:
            # Lê os dados do arquivo binário
            pc = np.fromfile(filename, dtype=np.float32)
            # Redimensiona os dados para uma matriz com num_points linhas e 3 colunas
            point_cloud = pc.reshape((-1, self.num_features))
            return point_cloud
        
        except Exception as e:
            print(f"Erro ao carregar nuvem de pontos do arquivo {filename}: {e}")
            return None
    
    
    def update_extracted_fog_file_list(self) -> None:

        if self.file_list is not None and 'extraction' not in self.file_name:
        # Obtém a lista de arquivos de névoa extraída
            self.extracted_fog_file_list = get_extracted_fog_file_list(FOG)

    def load_fog_points(self, index: int = None) -> None:
        # Verifica se a lista de arquivos de névoa extraída está vazia ou se o índice fornecido é inválido
        if not self.extracted_fog_file_list or (index is not None and (index < 0 or index >= len(self.extracted_fog_file_list))):
            print("A lista de arquivos de névoa extraída está vazia ou o índice fornecido é inválido.")
            return None

        # Se o índice não foi fornecido, gera um número aleatório para selecionar um arquivo da lista
        if index is None:
            index = RNG.integers(low=0, high=len(self.extracted_fog_file_list), size=1)[0]

        filename = self.extracted_fog_file_list[index]
        fog_points = np.fromfile(filename, dtype=self.d_type)


        return fog_points.reshape((-1, 5))



#metodo para salvar a point cloud
    def set_output_file(self, output_file: str) -> None:
        self.output_file = output_file


    def save_pointcloud_bin(self, point_cloud: np.ndarray, filename: str) -> None:
        # Salvando os dados da nuvem de pontos em um arquivo binário
        with open(filename, "wb") as f:
            # Escreve o número de pontos como um inteiro de 32 bits (4 bytes)
            f.write(np.array([point_cloud.shape[0]], dtype=np.int32).tobytes())
            f.write(point_cloud.astype(np.float32).tobytes())

        print(f"Nuvem de pontos salva em: {filename}")


        
loader = CloudLoader()
loader.load_clouds_from_workspace("/home/garmus/LiDAR_fog_sim/data/0000000000.bin")  # Aqui você especifica o caminho do arquivo que deseja carregar

# Carregar a nuvem de pontos
point_cloud = loader.load_pointcloud(loader.file_list[0])  # Carregar o primeiro arquivo da lista de arquivos
if point_cloud is not None:
    print("Nuvem de pontos carregada com sucesso:")
    print(point_cloud)
else:
    print("Falha ao carregar nuvem de pontos.")

# Simular a névoa na nuvem de pontos
pc_with_fog, simulated_fog_points, info_dict = simulate_fog(loader.p, point_cloud, loader.noise, loader.gain, loader.noise_variant)

# Salvar a nuvem de pontos com névoa
output_filename = "/home/garmus/Documents/fog.bin"
loader.save_pointcloud_bin(pc_with_fog, output_filename)

print("Matriz da nuvem de pontos com névoa:")
print(pc_with_fog)
