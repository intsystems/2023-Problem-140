import torch


ACC, LOSS, LAT = {'accuracy'}, {'loss'}, {'latency'}
ALL = ACC | LOSS | LAT

@torch.no_grad()
def validate(model, getters: set=ALL, *, dl, loss_fn, times, device):
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


def add_report(reports, discrete, backbone, **validate_kwargs):
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
    result = [0.] * n
    result[i] = 1
    return torch.tensor(result)
