import torch


ACC, LOSS, LAT = {'accuracy'}, {'loss'}, {'latency'}
ALL = ACC | LOSS | LAT

@torch.no_grad()
def validate(model, getters: set=ALL, *, dl, loss_fn=None, times=None, device):
    """
    :param model:
    :param getters: subset of {'accuracy','loss','latency'}
    :param dl: valid dataloader
    :param loss_fn: used to calculate loss
    :param times: used to calculate latency of backbone
    :param device:
    :return: average accuracy, loss and latency if corresponding getter is passed
    """
    n_true, n_tot = 0, 0
    loss, latency = 0, 0

    for i, (X, y) in enumerate(dl):
        if X.shape[0] != 64:
            continue

        y_pred = model(X.to(device))

        n_tot += 1

        if 'accuracy' in getters:
            n_true += (y_pred.argmax(-1) == y.to(device)).sum().item() / 64
        if 'loss' in getters:
            loss += loss_fn(y_pred, y.to(device)).item()
        if 'latency' in getters:
            latency += model.sample_gammas(previous=True).dot(times).item()

    return n_true / n_tot, loss / n_tot, latency / n_tot


def train_epoch(hypernet, backbone, optimizer, *, sample_lambd, dl, loss_fn, times, device):
    """
    :param hypernet:
    :param backbone:
    :param optimizer:
    :param sample_lambd:
    :param dl: train dataloader
    :param loss_fn:
    :param times:
    :param device:
    """ 
    for i, (X, y) in enumerate(dl):
        if X.shape[0] != 64:
            continue

        lambd = sample_lambd()

        optimizer.zero_grad()

        backbone.gammas = hypernet(lambd)
        y_pred = backbone(X.to(device))
        loss = loss_fn(y_pred, y.to(device)) + lambd * backbone.sample_gammas().dot(times)
        loss.backward()

        optimizer.step()


def add_report(reports, discrete: bool, backbone, **validate_kwargs):
    """
    Adds validate result to reports

    :param reports: dicts with keys 'acc', 'loss', 'lat' and values [...]
    :param discrete: validating on discrete backbone if True
    :param backbone: backbone model
    :param **validate_kwargs: passed to validate function after backbone
    """ 
    if discrete:
        backbone.make_gammas_discrete()

    acc, loss, lat = validate(backbone, **validate_kwargs)

    for report in reports:
        report['acc'].append(acc)
        report['loss'].append(loss)
        report['lat'].append(lat)

    if discrete:
        backbone.make_gammas_relaxed()


def one_hot(i: int, n: int):
    """
    one_hot(2, 4) returns tensor([[ 0., 0., 1., 0])

    :param i: position of 1
    :param n: size of vector
    :return: one hot vector filled with 0. and single 1.
    """ 
    result = [0.] * n
    result[i] = 1
    return torch.tensor(result)
