import numpy as np
import torch
import pickle

from mylib.utils import validate, one_hot


class GreedyAlgo: 
    def __init__(self, path):
        self.path = path

        try:
            self.cache = self.load_cache()
            print('loaded cache')
        except:
            self.cache = {}

        def fake_tqdm(gen, *args, **krwargs):
            """imitates api of tqdm w/o any visuals"""
            for value in gen:
                yield value

        self.tqdm = fake_tqdm

    def fit(self, backbone, full_report, lambd, *, return_history=False, acc_threshold=0.15, valid_params):
        backbone.make_gammas_discrete()

        greedy_gammas = torch.tensor([1.] * backbone.gammas.numel()).to(valid_params['device'])

        greedy_accs = [full_report['acc']]
        greedy_loss = [full_report['loss'] + lambd]
        greedy_lats = [full_report['lat']]

        for _ in self.tqdm(range(backbone.gammas.numel())):
            if greedy_accs[-1] < acc_threshold:
                continue

            greedy_gammas, acc, loss, lat = self.greedy_epoch(greedy_gammas, backbone, lambd, **valid_params)
            
            greedy_accs.append(acc)
            greedy_loss.append(loss)
            greedy_lats.append(lat)

        backbone.make_gammas_relaxed()

        if return_history:
            return greedy_accs, greedy_loss, greedy_lats
        else:
            best = np.argmin(greedy_loss)
            return greedy_accs[best], greedy_loss[best], greedy_lats[best]

    def load_cache(self, path=None):
        if path is None:
            path=self.path
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_cache(self, path=None, cache=None):
        if path is None:
            path=self.path
        if cache is None:
            cache=self.cache
        with open(path, 'rb') as f:
            pickle.dump(cache, f)

    def get_greedy_report(self, backbone, **valid_params):
        gammas = str(backbone.gammas)
        if gammas in self.cache:
            return self.cache[gammas]

        acc, loss, lat = validate(backbone, **valid_params)
        self.cache[gammas] = (acc, loss, lat)
        self.save_cache()

        return acc, loss, lat


    def greedy_epoch(self, greedy_gammas, backbone, lambd, **valid_params):
        reports = {
            'acc': [],
            'loss': [],
            'lat': [],
            'gammas': [],
        }

        for i in range(greedy_gammas.numel()):
            if greedy_gammas[i].item() == 1.0:
                backbone.gammas = greedy_gammas - one_hot(i, greedy_gammas.numel()).to(valid_params['device'])
                acc, loss, lat = self.get_greedy_report(backbone, **valid_params)

                reports['acc'].append(acc)
                reports['loss'].append(loss + lambd*lat)
                reports['lat'].append(lat)
                reports['gammas'].append(backbone.gammas.clone().detach())

        assert len(reports['loss']) > 0
        best = np.argmin(reports['loss'])
        return reports['gammas'][best], reports['acc'][best], reports['loss'][best], reports['lat'][best]
