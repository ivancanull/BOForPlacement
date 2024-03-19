import os
import argparse
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def parse_bo_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lb', type=int, help='resolution lower bound')
    parser.add_argument('--ub', type=int, help='resolution upper bound')
    parser.add_argument('--iter', type=int, default=10, help='number of iterations to run')
    parser.add_argument('--load', action='store_true', help='load previous runs and continue optimization')
    return parser.parse_args()
    

def objective_function(resolution_x, resolution_y):
    resolution_x, resolution_y = int(resolution_x), int(resolution_y)

    test_func = np.exp(-(resolution_x - 2) ** 2) + np.exp(-(resolution_x - 6) ** 2 / 10) + 1/ (resolution_y ** 2 + 1)

    return HPWL
    # TODO: i/o funtion for the placer

def plot_test_func(xlim, ylim):

    ax = plt.figure().add_subplot(projection='3d')
    X = np.arange(xlim[0], xlim[1], 0.25)
    Y = np.arange(ylim[0], ylim[1], 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.exp(-(X - 2) ** 2) + np.exp(-(X - 6) ** 2 / 10) + 1/ (Y ** 2 + 1)

    # plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,alpha=0.3)
    max_Z, min_Z = np.max(Z), np.min(Z)
    # plot projections of the contours for each dimension
    plot_zlim = (min_Z - 0.5 * (max_Z - min_Z), max_Z + 0.5 * (max_Z - min_Z))
    plot_xlim = (xlim[0] - 0.5 * (xlim[1] - xlim[0]), xlim[1] + 0.5 * (xlim[1] - xlim[0]))
    plot_ylim = (ylim[0] - 0.5 * (ylim[1] - ylim[0]), ylim[1] + 0.5 * (ylim[1] - ylim[0]))
    

    ax.contourf(X, Y, Z, zdir='z', offset=plot_zlim[0], cmap='coolwarm')
    ax.contourf(X, Y, Z, zdir='x', offset=plot_xlim[0], cmap='coolwarm')
    ax.contourf(X, Y, Z, zdir='y', offset=plot_ylim[1], cmap='coolwarm')

    ax.set(xlim=plot_xlim, ylim=plot_ylim, zlim=plot_zlim,
        xlabel='X', ylabel='Y', zlabel='Z')

    plt.savefig("./bo_test_func.png")

def main():
    # parse arguments
    args = parse_bo_args()

    resolution_lb, resolution_ub = args.lb, args.ub
    n_iter = args.iter
    load = args.load
    # define upper and lower boundaries
    pbounds = {'resolution_x': (resolution_lb, resolution_ub), 'resolution_y': (resolution_lb, resolution_ub)}
    
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    # load previous optimization results
    if os.path.exists("./bologs.json") and load:
        load_logs(optimizer, logs=["./bologs.json"])
        print("The optimizer is now aware of {} points.".format(len(optimizer.space)))
        init_points = 0 # must be set to 0
    else:
        init_points = 2 # how many steps of random exploration you want to perform

    # define logger to save the optimization progress
    logger = JSONLogger(path="./bologs")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=init_points, 
        n_iter=n_iter # how many steps of bayesian optimization you want to perform
    )

    # plot_test_func(xlim=(resolution_lb, resolution_ub),
    #                 ylim=(resolution_lb, resolution_ub))


if __name__ == '__main__':
    main()