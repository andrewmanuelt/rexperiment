import os 
import torch

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '24'

from util.dataset import DatasetUtility
from util.general import GeneralUtility
from util.evaluator import ProccessEvaluator

from config.hyperparameter import hyperparameter_config

def device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    print(f"Some of code running on", device)

def main():
    # Processing dataset
    du = DatasetUtility()
    du.run()
    
    # Get dataset profile > content_length.json
    gu = GeneralUtility()
    gu.get_max_length()
    
    # Main hyperparameter
    # config = hyperparameter_config()
    # pe = ProccessEvaluator()
    # pe.hyperparameter_grid(config)

if __name__ == '__main__':
    device()
    main()