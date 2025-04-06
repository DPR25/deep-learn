from learn import *
from data import get_pair_paths, ForestDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':

    DATASET_NAME = 'ATLANTIC'
    EPOCHS = 40
    MODEL_NAME = 'Jimmy'

    # data
    train_pairs = get_pair_paths(f'{DATASET_NAME}/Training/image', f'{DATASET_NAME}/Training/label') 
    train_dataset = ForestDataset(train_pairs)
    train_loader = DataLoader(train_dataset, batch_size=8)

    # model
    print('Loading pretrained backbone ...')
    model = get_model('cuda')

    # training loop
    forest_finetune(model, train_loader, EPOCHS, MODEL_NAME, 1e-4, 'cuda')
  
