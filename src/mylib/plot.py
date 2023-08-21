import matplotlib.pyplot as plt

def convergence(lambd_grid, *args, saveto):
    plt.figure(figsize=(4*len(lambd_grid), 3*len(args)))
    for j, (report, label) in enumerate(args):
        for i, lambd in enumerate(lambd_grid, 1):
            plt.subplot(len(args), len(lambd_grid), i + j*len(lambd_grid))
            plt.set_title(rf"$\lambda={lambd}$")
            plt.plot(report[lambd], label=label)
            plt.legend()
    plt.savefig(saveto, bbox_inches='tight')

def vs_lambd(lambd_grid, *args, saveto):
    plt.figure(figsize=(4*len(args), 3))
    for i, (values, base, label) in enumerate(args, 1):
        plt.subplot(1, len(args), i)
        plt.xticks(lambd_grid)
        plt.plot(lambd_grid, [base] * len(lambd_grid), label='base', ls=':')
        plt.plot(lambd_grid, values, label=label)
        plt.xlabel(r'$\lambda$')
        plt.legend()
    plt.savefig(saveto, bbox_inches='tight')
