import torch

ACC, LOSS, LAT = {'accuracy'}, {'loss'}, {'latency'}

@torch.no_grad()
def validate(model, dataloader, getters, *, loss_fn=None, times=None, device='cpu'):
    n_true, n_tot = 0, 0
    loss, latency = 0, 0

    if 'loss' in getters:
        assert loss_fn != None, '`loss_fn` is required to collect loss'
    if 'latency' in getters:
        assert times != None, '`times` is required to collect loss'

    for X, y in dataloader:
        if X.shape[0] != 64:
            continue
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        if 'accuracy' in getters:
            n_true += (y_pred.argmax(-1) == y).sum().item()
        if 'loss' in getters:
            if loss_fn.reduction == 'mean':
                loss += loss_fn(y_pred, y).item() * X.shape[0]
            elif loss_fn.reduction == 'sum':
                loss += loss_fn(y_pred, y).item()
            else:
                assert False, f'use "mean" or "sum" as {loss_fn}.reduction, not {loss_fn.reduction}'
        if 'latency' in getters:
            latency += model.sample_gammas(previous=True).dot(times).item() * X.shape[0]
        n_tot += X.shape[0]

    return n_true / n_tot, loss / n_tot, latency / n_tot

