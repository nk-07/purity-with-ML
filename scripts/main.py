import torch
import torch.nn as nn
from utils import *
from create_dataset import MultiTaskDataset
from LibMTL import Trainer
from LibMTL.config import LibMTL_args, prepare_args
from torch.utils.data import DataLoader


def parse_args(parser):
    parser.add_argument('--train_bs', default=60, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=60, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=60, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='./DB/', type=str, help='dataset path')
    return parser.parse_args()
    
def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    scheduler_param = {
    'scheduler': 'reduce',
    'mode': 'max',          
    'factor': 0.9,          # Gentle reduction (10%)
    'patience': 5,          # Wait 5 epochs without improvements
    'cooldown': 2,          # Pause after LR change
    'threshold': 0.001,     # Minimum significant improvement
    'threshold_mode': 'rel' # Relative change (best_metric * (1 - threshold))
    }

    # prepare dataloaders
    train_dataset = MultiTaskDataset(data_dir=params.dataset_path, mode='train')
    val_dataset = MultiTaskDataset(data_dir=params.dataset_path, mode='val')
    test_dataset = MultiTaskDataset(data_dir=params.dataset_path, mode='test')
    
    Spectra_test_loader = DataLoader(test_dataset, batch_size=params.train_bs, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=params.test_bs, shuffle=False, num_workers=2, pin_memory=True)
    Spectra_train_loader = DataLoader(train_dataset, batch_size=params.test_bs, shuffle=True, num_workers=2, pin_memory=True)
    
    task_dict = {'fg': {'metrics':['F1'], 
                              'metrics_fn': MultiLabelF1Metric(),
                              'loss_fn': BCELoss(),
                              'weight': [1]}, 
                 'purity': {'metrics':['F1'], 
                            'metrics_fn': F1Metric(),
                            'loss_fn': MultiMarginLoss(),
                            'weight': [1]}}

    class IrCNN(nn.Module):
        def __init__(self, signal_size=600, kernel_size=4, in_ch=1, p=0.388079443185074):
            super(IrCNN, self).__init__()

            self.CNN1 = nn.Sequential(
                nn.Conv1d(in_channels=in_ch, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
                nn.BatchNorm1d(num_features=10),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.cnn1_size = int(((signal_size - kernel_size + 1 - 2) / 2) + 1)
            
            self.CNN2 = nn.Sequential(
                nn.Conv1d(in_channels=10, out_channels=20, kernel_size=kernel_size, stride=1, padding=0),
                nn.BatchNorm1d(num_features=20),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.cnn2_size = int(((self.cnn1_size - kernel_size + 1 - 2) / 2) + 1)

            self.CNN3 = nn.Sequential(
                nn.Conv1d(in_channels=20, out_channels=40, kernel_size=kernel_size, stride=1, padding=0),
                nn.BatchNorm1d(num_features=40),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2))

            self.cnn3_size = int(((self.cnn2_size - kernel_size + 1 - 2) / 2) + 1)

            self.DENSE1 = nn.Sequential(
                nn.Linear(in_features=self.cnn3_size * 40, out_features=1015),
                nn.ReLU(),
                nn.Dropout(p=p)
            )
            
            self.DENSE2 = nn.Sequential(
                nn.Linear(in_features=1015, out_features=733),
                nn.ReLU(),
                nn.Dropout(p=p)
            )
            
            self.DENSE3 = nn.Sequential(
                nn.Linear(in_features=733, out_features=529),
                nn.ReLU(),
                nn.Dropout(p=p)
            )

        def forward(self, signal):
            x = self.CNN1(signal)
            x = self.CNN2(x)
            x = self.CNN3(x)
            x = torch.flatten(x, -2, -1)
            x = self.DENSE1(x)
            x = self.DENSE2(x)
            x = self.DENSE3(x)
            return x
    
    def encoder_class(): 
        return IrCNN()

    num_out_channels = {'fg': 16, 'purity': 16}

    decoders = nn.ModuleDict({'fg': nn.Linear(529, num_out_channels['fg']),
        'purity': nn.Linear(529, num_out_channels['purity'])})

    
    class SpectraTrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class,
                     decoders, rep_grad, multi_input, optim_param, 
                     scheduler_param, **kwargs):
            super(SpectraTrainer, self).__init__(task_dict=task_dict, 
                                            weighting=weighting, 
                                            architecture=architecture, 
                                            encoder_class=encoder_class,
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            **kwargs)

    
    SpectraModel = SpectraTrainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=encoder_class,
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          **kwargs)
    if params.mode == 'train':
        SpectraModel.train(Spectra_train_loader, Spectra_test_loader, params.epochs,val_dataloaders=val_loader)
    elif params.mode == 'test':
        SpectraModel.test(Spectra_test_loader)
    else:
        raise ValueError

if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    main(params)
