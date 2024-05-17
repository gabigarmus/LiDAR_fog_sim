import os
import numpy as np 
from fog_simulation import ParameterSet, RNG, simulate_fog
from typing import List

def get_extracted_fog_file_list(bin_path: str) -> List[str]:
        file_list = [y for x in os.walk(bin_path) for y in glob(os.path.join(x[0], f'*.bin'))]
        print("Arquivos de névoa extraída encontrados:", file_list)
        return sorted(file_list)

def read_kitti_bin(bin_path: str):
        """Função para ler o arquivo .bin do dataset KITTI"""
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points

def save_kitti_bin(points_with_noise, bin_path):
    """Função para salvar a nuvem de pontos em formato .bin"""
    points_with_noise.tofile(bin_path, format='float64')
    print(f"Nuvem de pontos com ruído salva em {bin_path}")
    #print("Matriz da nuvem de pontos com ruído:")
    #print(points_with_noise)  # Adicionando o print da matriz
               
def main():
    noise = 10
    gain = True
    noise_variant ='v4'
    parameters = ParameterSet(n = 500,n_min = 100,n_max = 1000,r_range = 79.99797,r_range_min = 50,r_range_max = 250,alpha = 0.06,alpha_min = 0.003,alpha_max = 0.5,alpha_scale = 1000,mor = 49.92887122589985,beta = 0.000921310633919122,beta_min = 0.000460655316959561,beta_max = 0.001842621267838244,beta_scale = 49928.87122589985,p_0 = 80,p_0_min = 60,p_0_max = 100,tau_h = 2e-08,tau_h_min = 5e-09,tau_h_max = 8e-08,tau_h_scale = 1000000000.0,e_p = 1.6e-06,a_r = 0.25,a_r_min = 0.01,a_r_max = 0.1,a_r_scale = 1000,l_r = 0.05,l_r_min = 0.01,l_r_max = 0.1,l_r_scale = 100,c_a = 1873702.8625,linear_xsi = True,D = 0.1,ROH_T = 0.01,ROH_R = 0.01,GAMMA_T_DEG = 2,GAMMA_R_DEG = 3.5,GAMMA_T = 0.03490658503988659,GAMMA_R = 0.061086523819801536,r_1 = 0.9,r_1_min = 0,r_1_max = 10,r_1_scale = 10,r_2 = 1.0,r_2_min = 0,r_2_max = 10,r_2_scale = 10,r_0 = 30,r_0_min = 1,r_0_max = 200,gamma = 1e-06,gamma_min = 1e-07,gamma_max = 1e-05,gamma_scale = 10000000,beta_0 = 3.1830988618379064e-07)

    #bin_path = "/home/garmus/LiDAR_fog_sim/data/0000000000.bin"  # Insira o caminho para o arquivo .bin do dataset KITTI
    
    bin_path = "/home/garmus/datasets/KITTI/3D/training/velodyne/000000.bin"
    
    #bin_path = "/home/garmus/LiDAR_fog_sim/data/0.bin"
    output_bin_path = "/home/garmus/Documents/salva_cloud.bin"  # Insira o caminho onde deseja salvar o novo arquivo .bin

    # Carregando os pontos da nuvem de pontos
    #pc = np.fromfile(bin_path)
    #pc = pc.reshape((-1, 4))
    pc = read_kitti_bin(bin_path)
    intensity_multiplier = 255

    pc[:,3] = np.round(pc[:,3] * intensity_multiplier)

    # Simulando a névoa na nuvem de pontos
    augmented_pccloud, pc_with_fog, info_dict = simulate_fog(parameters, 
                                                             pc,
                                                             noise,
                                                             gain,
                                                             noise_variant)
    
    print(info_dict)
    # Salvar a nuvem de pontos com ruído em formato .bin
    save_kitti_bin(augmented_pccloud, output_bin_path)


if __name__ == "__main__":
    main()



