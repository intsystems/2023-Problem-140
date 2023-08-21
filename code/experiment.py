import argparse

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.fx

from torch.distributions.uniform import Uniform

import numpy as np
import pandas as pd


class PwHypernet(nn.Module):
    def __init__(self, n_intervals:int = 5, out_size: int = 10, *, Lambd):
        super().__init__()
        self.intervals = nn.Parameter(torch.randn(n_intervals, out_size))
        self.Lambd = Lambd
        
    def forward(self, lambd: torch.Tensor):
        assert 0 <= lambd and lambd < self.Lambd, f'lambd should be in [0, Lambd={self.Lambd})'

        idx = (lambd / self.Lambd * self.intervals.shape[0]).long().item()
        return self.intervals[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default='cpu',
                        help='device (default: cpu)')
    parser.add_argument("--figures", default='',
                        help='path to directory to save to, don\'t plot if omitted')
    parser.add_argument("--model",  default='pretrained/model_23.ckpt',
                        help='path to model (default: pretrained/model_23.ckpt)')
    parser.add_argument("--times",  default='time_measurements/ResNet18HomeMeasurements.csv',
                        help='path to times (deafult: time_measurements/ResNet18HomeMeasurements.csv)')
    parser.add_argument("--repo",  default='2023-Problem-140',
                        help='path to repo (deafult: 2023-Problem-140)')
    parser.add_argument("--dataset",  default='CIFAR10',
                        help='CIFAR10, CIFAR100, or ImageNet (deafult: CIFAR10)')
    parser.add_argument("--lambd", type=float, default=10.0,
                        help='upper bound for lambda such that the graph is connected')
    parser.add_argument("--epochs", type=int, default=25,
                        help='number of training epochs')
    parser.add_argument("--image-size", type=int, default=33,
                        help='desired output size of the crop (default: 33)')
    parser.add_argument("--init-shift", type=float, default=4.0)
    parser.add_argument("--delta-shift", type=float, default=0.5)
    parser.add_argument("--init-accuracy", type=float, default=0.4)
    args = parser.parse_args()

    device = args.device
    figures = args.figures
    path_to_model = args.model
    path_to_times = args.times
    path_to_repo = args.repo
    dataset = args.dataset
    Lambd = args.lambd
    epochs = args.epochs
    image_size = args.image_size
    init_shift = args.init_shift
    delta_shift = args.delta_shift
    init_accuracy = args.init_accuracy


    # mylib

    import sys
    sys.path.append(f'{path_to_repo}/src')

    from mylib.nas.module2graph import GraphInterperterWithGamma
    from mylib.dataloader import get_dataloaders
    from mylib.validate import validate, ACC, LOSS, LAT
    from mylib.model import ResNet18
    import mylib.plot as plot


    # interpreter

    class GraphInterperterWithGumbelSoftmaxGamma(GraphInterperterWithGamma):
        def __init__(self, mod, gamma_shift=0.0, temperature=1.0):
            self.gamma_shift = gamma_shift
            self.temperature = temperature
            super().__init__(mod)
    
        def init_gammas(self):
            i = 0
            gammas = []
            self.gammas_name = {}
            self.atomic_operations = []
            for node in self.graph.nodes:
                if node.op == 'call_module':
                    self.atomic_operations.append(node)
                    gammas.append(np.random.randn()+self.gamma_shift)
                    self.gammas_name[str(node)] = i# перевод в str тут для удобства. в реалньых методах это не нужно
                    i+=1                        # да и вообще, тут по идее должен быть тензор/параметр
            self.gammas = torch.as_tensor(gammas).to(device)
            self.discrete = False 
    
        def sample_gammas(self, previous=False):
            if self.discrete:
                return self.gammas
    
            if not previous:
                d = torch.distributions.RelaxedBernoulli(logits=self.gammas, temperature=self.temperature)
                self.last_sample = d.rsample()
    
            return self.last_sample
            
        def make_gammas_discrete(self):
            self.gammas.data = (self.gammas.data>0) * 1.0
            self.discrete = True
    
        def relax_gammas(self, gammas=None):
            if gammas is not None:
                self.gammas.data = gammas.data
            self.discrete = False


    # model

    model = ResNet18(dataset, path_to_model=path_to_model).to(device)

    # data & times

    train_dl, test_dl = get_dataloaders(dataset, img_size=image_size)

    times = pd.read_csv(path_to_times, index_col=0)
    times = torch.tensor(times['mean'], dtype=torch.float32).to(device)
    times /= times.sum()

    loss_fn = torch.nn.CrossEntropyLoss()

    # base (full graph)

    base_accuracy, base_loss, _ = validate(model, test_dl, ACC | LOSS, loss_fn=loss_fn, device=device)
    print('base:', base_accuracy, base_loss)


    # worst (epmty graph)

    imodel = GraphInterperterWithGumbelSoftmaxGamma(model.eval()).to(device).eval()

    imodel.gammas = torch.zeros_like(imodel.gammas).to(device)
    imodel.make_gammas_discrete()

    worst_accuracy, worst_loss, _ = validate(imodel, test_dl, ACC | LOSS, loss_fn=loss_fn, device=device)

    imodel.relax_gammas()

    print('worst:', worst_accuracy, worst_loss)


    # init gammas

    print('finding init gammas...')

    current_accuracy = worst_accuracy
    curr_shift = init_shift
    init_gammas = None

    while current_accuracy <= init_accuracy:
        imodel.gammas = torch.randn(*imodel.gammas.shape).to(device) + curr_shift
        current_accuracy, _, _ = validate(imodel, test_dl, ACC, device=device)

        print(current_accuracy, curr_shift)
        
        curr_shift += delta_shift

    init_gammas = imodel.gammas
    init_gammas.sigmoid()


    # init hypernet

    n_intervals = 10

    hypernet = PwHypernet(n_intervals=n_intervals, out_size=imodel.gammas.numel(), Lambd=Lambd).to(device)
    optimizer = torch.optim.Adam(hypernet.parameters(), lr=2e-2)

    hypernet.intervals.data = hypernet.intervals.data * 0 + init_gammas[None].data

    lambd = torch.tensor(0.0).to(device)
    assert torch.all(hypernet(lambd) == hypernet(lambd + 0.5))


    # training

    lambda_sampler = Uniform(0, Lambd)


    def train_epoch(imodel, hypernet, optimizer, lambda_sampler=lambda_sampler, dl=train_dl, temperature=1):
        for i, (X, y) in enumerate(dl):
            if X.shape[0] != 64:
                continue

            lambd = lambda_sampler.sample().to(device).view(1)

            optimizer.zero_grad()

            imodel.gammas = hypernet(lambd)
            y_pred = imodel(X.to(device))
            loss = loss_fn(y_pred, y.to(device)) + lambd * imodel.sample_gammas(previous=False).dot(times)
            loss.backward()

            optimizer.step()

    temperature = 0.3

    lambd_grid = [i*Lambd/n_intervals for i in range(n_intervals)]

    lambda_report = {
        'acc':{str(lambd):[] for lambd in lambd_grid},
        'lat':{str(lambd):[] for lambd in lambd_grid},
    }

    for epoch in tqdm(range(epochs), desc='training', total=epochs):
        hypernet.train()

        train_epoch(imodel, hypernet, optimizer, temperature=temperature)
        temperature -= 0.05
        temperature = max(0.15, temperature)

        hypernet.eval()

        with torch.no_grad():
            for lambd in range(n_intervals):
                lambd *= Lambd / n_intervals
                imodel.gammas = hypernet(torch.tensor(lambd))

                acc, loss, lat = validate(imodel, test_dl, ACC | LOSS | LAT, loss_fn=loss_fn, times=times, device=device)

                lambda_report['acc'][str(lambd)].append(acc)
                lambda_report['lat'][str(lambd)].append(lat)

    print('report', lambda_report)
    

    # collect vs lambda

    acc_vs_lambda = [r[-1] for _, r in lambda_report['acc'].items()]
    lat_vs_lambda = [r[-1] for _, r in lambda_report['lat'].items()]

    print('lambda grid', lambd_grid)
    print('acc vs lambda', acc_vs_lambda)
    print('lat vs lambda', lat_vs_lambda)

    # plot

    if figures:
        plot.convergence(lambd_grid, (lambda_report['acc'], 'acc'), (lambda_report['lat'], 'lat'), saveto=f'{figures}/convergence.pdf')
        plot.vs_lambd(lambd_grid, (acc_vs_lambda, base_accuracy, 'acc'), (lat_vs_lambda, 1, 'lat'), saveto=f'{figures}/vs_lambd.pdf')
