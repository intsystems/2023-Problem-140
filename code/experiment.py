import argparse

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.fx

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns
# mpl.style.use('seaborn')


class HyperNet(nn.Module):
    def __init__(self, out_size, hidden_layer_num=1, hidden_size=128):
        """
        :param hidden_layer_num: количество скрытых слоев (может быть нулевым)
        :param hidden_size: количество нейронов на скрытом слое (актуально, если скрытые слои есть)
        :param out_size: количество параметров
        """
        nn.Module.__init__(self)

        layers = []
        in_ = 1 # исходная входная размерность
        for _ in range(hidden_layer_num):
            layers.append(nn.Linear(in_, hidden_size))
            layers.append(nn.ReLU())
            in_ = hidden_size
        layers.append(nn.Linear(in_, out_size))

        self.model = nn.Sequential(*layers)

    def forward(self, lambd):
        return self.model(lambd)


class PwHypernet(nn.Module):
    def __init__(self, n_intervals:int = 5, out_size: int = 10, *, Lambd):
        super().__init__()
        self.intervals = nn.Parameter(torch.randn(n_intervals, out_size))
        self.Lambd = Lambd
        
    def forward(self, lambd: torch.Tensor):
        assert 0 <= lambd and lambd < self.Lambd, f'lambd should be in [0, Lambd={self.Lambd})'

        idx = (lambd / self.Lambd * self.intervals.shape[0]).long().item()
        return self.intervals[idx]


loss_fn = torch.nn.CrossEntropyLoss()
ACC, LOSS, LAT = {'accuracy'}, {'loss'}, {'latency'}
ALL = ACC | LOSS | LAT
@torch.no_grad()
def validate(model, dataloader, device, getters: set=ALL):
    n_true, n_tot = 0, 0
    loss, latency = 0, 0

    for i, (X, y) in enumerate(dataloader):
        if X.shape[0] != 64:
            continue

        y_pred = model(X.to(device))

        n_tot += 64

        if 'accuracy' in getters:
            n_true += (y_pred.argmax(-1) == y.to(device)).sum().item()
        if 'loss' in getters:
            loss += loss_fn(y_pred, y.to(device)).item() * 64
        if 'latency' in getters:
            latency += model.sample_gammas(previous=True).dot(times).item() * 64

    return n_true / n_tot, loss / n_tot, latency / n_tot


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
    args = parser.parse_args()

    device = args.device
    figures = args.figures
    path_to_model = args.model
    path_to_times = args.times
    path_to_repo = args.repo


    # mylib

    import sys
    sys.path.append(f'{path_to_repo}/src')

    from mylib.nas.resnet18 import ResNet18
    from mylib.nas.module2graph import GraphInterperterWithGamma
    from mylib.nas.cifar_data import get_dataloaders


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

    model = ResNet18(num_classes=10).to(device)
    model.load_state_dict(torch.load(path_to_model))


    # data & times

    train_dl, test_dl = get_dataloaders(classes=range(10), batch_size=64, img_size=33)

    times = pd.read_csv(path_to_times, index_col=0)
    times = torch.tensor(times['mean'], dtype=torch.float32).to(device)
    times /= times.sum()


    # base (full graph)

    base_accuracy, base_loss, _ = validate(model.to(device), test_dl, device, ACC | LOSS)
    print('base:', base_accuracy, base_loss)


    # worst (epmty graph)

    imodel = GraphInterperterWithGumbelSoftmaxGamma(model.eval()).to(device).eval()

    imodel.gammas = torch.zeros_like(imodel.gammas).to(device)
    imodel.make_gammas_discrete()

    worst_accuracy, worst_loss, _ = validate(imodel, test_dl, device, ACC | LOSS)

    imodel.relax_gammas()

    print('worst:', worst_accuracy, worst_loss)


    # init gammas

    print('finding init gammas...')

    current_accuracy = worst_accuracy
    curr_shift = 4.0
    init_gammas = None

    while current_accuracy <= 0.4:
        imodel.gammas = torch.randn(*imodel.gammas.shape).to(device) + curr_shift
        current_accuracy, _, _ = validate(imodel, test_dl, device, ACC)

        print(current_accuracy, curr_shift)
        
        curr_shift += 0.5
    curr_shift -= 0.5

    init_gammas = imodel.gammas
    init_gammas.sigmoid()


    # init hypernet

    Lambd = 10.0
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

    EPOCHS = 20

    temperature = 0.3

    lambda_report = {}

    for epoch in tqdm(range(EPOCHS), desc='training', total=EPOCHS):
        hypernet.train()

        train_epoch(imodel, hypernet, optimizer, temperature=temperature)
        temperature -= 0.05
        temperature = max(0.15, temperature)

        hypernet.eval()

        with torch.no_grad():
            for lambd in range(n_intervals):
                lambd *= Lambd / n_intervals
                imodel.gammas = hypernet(torch.tensor(lambd))

                acc, loss, latency = validate(imodel, test_dl, device)

                lambda_report.setdefault(f'{lambd}', {})
                lambda_report[f'{lambd}'].setdefault('acc', []).append(acc)
                lambda_report[f'{lambd}'].setdefault('latency', []).append(latency)

    print('report', lambda_report)
    

    # collect vs lambda

    accuracy_vs_lambda = []
    latency_vs_lambda = []

    lambd_grid = [i*Lambd/n_intervals for i in range(n_intervals)]

    for lambd in lambd_grid:
        imodel.gammas = hypernet(torch.tensor(lambd))
        imodel.make_gammas_discrete()
        accuracy, _, latency = validate(imodel, test_dl, device)

        latency_vs_lambda.append(latency)
        accuracy_vs_lambda.append(accuracy)

    
    print('acc vs lambda', accuracy_vs_lambda)
    print('lat vs lambda', latency_vs_lambda)

    # plot

    if figures:
        # convergence
        fig, axs = plt.subplots(2, 5, figsize=(20, 6))
        for i, key in enumerate(lambda_report):
            h, w = i//5, i%5
            axs[h][w].set_title(rf"$\lambda={key}$")
            axs[h][w].plot([base_accuracy]*EPOCHS, label='base', ls=':')
            axs[h][w].plot(lambda_report[key]['acc'], label='acc')
            axs[h][w].plot(lambda_report[key]['latency'], label='lat')
            axs[h][w].legend()
            axs[h][w].set_ylim(0, 1)
        plt.savefig(f'{figures}/convergence.pdf', bbox_inches='tight')

        # vs lambda

        plt.figure(figsize=(5,3))
        plt.xticks(list(range(10)))
        plt.plot(lambd_grid, [base_accuracy] * len(lambd_grid), label='base', ls=':')
        plt.plot(lambd_grid, accuracy_vs_lambda, label='acc')
        plt.plot(lambd_grid, latency_vs_lambda, label='lat')
        plt.legend()
        plt.xlabel(r'$\lambda$')
        plt.savefig(f'{figures}/lat&acc-lambd.pdf', bbox_inches='tight')
