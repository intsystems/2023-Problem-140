import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
import torch

from mylib.utils import add_report, train_epoch, validate
from mylib.hypernet import ConstHypernet


class BasicExperiment:
    def __init__(self, backbone, *, lambds):
        self.backbone = backbone
        self.init_gammas = self.backbone.gammas
        self.lambds = lambds

    def conduct(self, *, temperature=0.2, lr=0.01, epochs=15, desc='', valid_params, train_params):
        relaxed_report  = {lambd: {'acc': [], 'lat': [], 'loss': []} for lambd in self.lambds}
        discrete_report = {lambd: {'acc': [], 'lat': [], 'loss': []} for lambd in self.lambds}

        self.backbone.gammas = self.init_gammas
        self.backbone.temperature = temperature

        add_report(relaxed_report.values(), False, self.backbone, **valid_params)
        add_report(discrete_report.values(), True, self.backbone, **valid_params)

        for lambd in tqdm_notebook(self.lambds, total=len(self.lambds), desc=desc):
            hypernet = ConstHypernet(init_gammas=self.init_gammas.clone().detach())
            optimizer = torch.optim.Adam(hypernet.parameters(), lr=lr)

            for epoch in tqdm_notebook(range(epochs), total=epochs, desc=f'lambd={lambd}'):
                hypernet.train()
                train_epoch(hypernet, self.backbone, optimizer, sample_lambd=lambda : lambd, **train_params)
                hypernet.eval()

                self.backbone.gammas = hypernet(lambd)
                add_report([relaxed_report[lambd]], False, self.backbone, **valid_params)
                add_report([discrete_report[lambd]], True, self.backbone, **valid_params)

        return relaxed_report, discrete_report
    
    def validate(self, backbone: str, valid_params):
        """
        :param backbone: "current", "full", "empty"
        """

        assert backbone in {"current", "full", "empty"}

        backbone_gammas = self.backbone.gammas.copy()

        if backbone == "current":
            pass
        if backbone == "full":
            self.backbone.gammas *= 0
            self.backbone.gammas += 1
            self.backbone.make_gammas_discrete()
        if backbone == "empty":
            self.backbone.gammas *= 0
            self.backbone.make_gammas_discrete()
        
        acc, loss, lat = validate(self.backbone, **valid_params)

        self.backbone.gammas = backbone_gammas
        return acc, loss, lat


    def plot_convergence_single(self, history, base, *, epochs=15, ylim, ylim_margin=0.05):
        plt.plot([base]*(epochs+1), label='full', ls=':')
        plt.plot(history, label='basic')
        plt.ylim(ylim[0] - ylim_margin, ylim[1] + ylim_margin)


    def plot_convergence(self, report, full_report, empty_report, tag: str,
                         epochs=15, save_path=None):
        w = 4
        plt.figure(figsize=(w*4, 6))

        for i, (lambd, report) in enumerate(report.items(), 1):
            plt.subplot(3, w, i)
            plt.title(rf"$\lambda={lambd}$")
            self.plot_convergence_single(report['acc'], full_report['acc'], ylim=(empty_report['acc'], full_report['acc']))
            if i == 1:
                plt.ylabel('accuracy')
            if i == 4:
                plt.legend()

            plt.subplot(3, w, i + w)
            self.plot_convergence_single(report['lat'], full_report['lat'], ylim=(empty_report['lat'], full_report['lat']))
            if i == 1:
                plt.ylabel('latency')
            if i == 4:
                plt.legend()

            plt.subplot(3, w, i + 2*w)
            loss_history = [loss + lambd*lat for loss, lat in zip(report['loss'], report['lat'])]
            plt.plot([empty_report['loss'] + lambd]*(epochs+1), label='worst w/ regularization', ls=':')
            plt.plot([full_report['loss']]*(epochs+1), label='full w/o regularization', ls=':')
            plt.plot(loss_history, label='basic')
            if i == 1:
                plt.ylabel('loss + $\lambda\cdot$latency')
            if i == 4:
                plt.legend()
            plt.xlabel('epoch')

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')


    def plot_vs_lambda_single(self, values, base, *, ylim, ylim_margin=0.05, color):
        plt.plot(self.lambds, [base] * len(values), label='base', ls=':')
        plt.plot(self.lambds, values, label='basic', color=color)
        plt.xticks(self.lambds)
        plt.ylim(ylim[0]-ylim_margin, ylim[1]+ylim_margin)
        plt.legend()

    def plot_vs_lambda(self, relaxed_report, discrete_report, full_report, empty_report, tag: str,
                       save_path=None):
        accs = [r['acc'][-1] for l, r in relaxed_report.items()]
        lats = [r['lat'][-1] for l, r in relaxed_report.items()]

        plt.figure(figsize=(6,6))

        plt.subplot(2, 2, 1)
        plt.title(f'relaxed {tag}')
        self.plot_vs_lambda_single(accs, full_report['acc'], ylim=(empty_report['acc'], full_report['acc']), color='C1')
        plt.ylabel('accuracy')

        plt.subplot(2, 2, 3)
        self.plot_vs_lambda_single(lats, full_report['lat'], ylim=(empty_report['lat'], full_report['lat']), color='C2')
        plt.ylabel('latency')

        accs = [r['acc'][-1] for l, r in discrete_report.items()]
        lats = [r['lat'][-1] for l, r in discrete_report.items()]

        plt.subplot(2, 2, 2)
        plt.title(f'discrete {tag}')
        self.plot_vs_lambda_single(accs, full_report['acc'], ylim=(empty_report['acc'], full_report['acc']), color='C1')

        plt.subplot(2, 2, 4)
        self.plot_vs_lambda_single(lats, full_report['lat'], ylim=(empty_report['lat'], full_report['lat']), color='C2')

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')


    def plot_abl1(self, report, gaccs, glats, tag: str,
                  exclude_points:set={}, exclude_points_greedy:set={3,4, 6,7},
                  text_offset=(-0.01, -0.06), save_path=None):
        accs = [r['acc'][-1] for l, r in report.items()]
        lats = [r['lat'][-1] for l, r in report.items()]

        plt.figure(figsize=(5,4))

        plt.scatter(lats, accs, label='basic')
        plt.plot(lats, accs)

        plt.scatter(glats, gaccs, label='greedy')
        plt.plot(glats, gaccs, ls='--')

        for i, (lat, acc) in enumerate(zip(lats, accs)):
            if i in exclude_points:
                continue
            plt.text(lat+text_offset[0], acc+text_offset[1], rf'$\lambda={self.lambds[i]}$', color='C0')

        for i, (lat, acc) in enumerate(zip(glats, gaccs)):
            if i in exclude_points_greedy:
                continue
            plt.text(lat+text_offset[0], acc+text_offset[1], rf'$\lambda={i}$', color='C1')

        plt.xlabel('latency')
        plt.ylabel('accuracy')

        plt.xlim((0,1))
        plt.ylim((0,0.8))

        plt.legend()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')


    def plot_abl2(self, relaxed_report, discrete_report, tag,
                  exclude_points: set={}, text_offset=(0.01, -0.02),
                  value_tag='acc', value_label='accuracy', save_path=None):
        plt.figure(figsize=(4,4))

        diss = [report[value_tag][-1] for _, report in discrete_report.items()]
        rels = [report[value_tag][-1] for _, report in relaxed_report.items()]

        worst_acc = min(min(diss), min(rels))
        best_acc = max(max(diss), max(rels))

        plt.plot([worst_acc, best_acc], [worst_acc, best_acc], ls='--', color='green', label='id')
        plt.scatter(diss, rels, label='actual')
        plt.plot(diss, rels)

        for i, (dis, rel) in enumerate(zip(diss, rels)):
            if i in exclude_points:
                continue
            plt.text(dis+text_offset[0], rel+text_offset[1], rf'$\lambda={self.lambds[i]}$')

        plt.xlabel(f'{value_label} (discrete model)')
        plt.ylabel(f'{value_label} (relaxed model)')

        plt.legend()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
