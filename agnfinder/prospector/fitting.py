import numpy as np

def get_best_theta(model, output):
    results = output["optimization"][0]
    # Several minimisations happen, according to nmin
    # Find which of the minimizations gave the best result
    index_of_best = np.argmin([r.cost for r in results])
    print('Index of best result: {}'.format(index_of_best))
    theta_best = results[index_of_best].x.copy()
    print('Best parameters: {}'.format(dict(zip(model.free_params, theta_best))))
    return theta_best
