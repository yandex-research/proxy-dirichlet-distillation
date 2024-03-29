import os

import numpy as np
from scipy.special import softmax


def load_uncertainties(path, n_best=5, beam_width=5, beam_search=True, temp=1.0):
    eoe = np.loadtxt(os.path.join(path, 'entropy_expected.txt'), dtype=np.float64)
    exe = np.loadtxt(os.path.join(path, 'expected_entropy.txt'), dtype=np.float64)
    mi = np.loadtxt(os.path.join(path, 'mutual_information.txt'), dtype=np.float64)
    epkl = np.loadtxt(os.path.join(path, 'epkl.txt'), dtype=np.float64)
    mkl = np.loadtxt(os.path.join(path, 'mkl.txt'), dtype=np.float64)
    score = np.loadtxt(os.path.join(path, 'score.txt'), dtype=np.float64)
    aep_tu = np.loadtxt(os.path.join(path, 'aep_tu.txt'), dtype=np.float64)
    aep_du = np.loadtxt(os.path.join(path, 'aep_du.txt'), dtype=np.float64)
    npmi = np.loadtxt(os.path.join(path, 'npmi.txt'), dtype=np.float64)
    score_npmi = np.loadtxt(os.path.join(path, 'score_npmi.txt'), dtype=np.float64)
    lprobs = np.loadtxt(os.path.join(path, 'log_probs.txt'), dtype=np.float64)

    ep_eoe = np.loadtxt(os.path.join(path, 'ep_entropy_expected.txt'), dtype=np.float64)
    ep_mi = np.loadtxt(os.path.join(path, 'ep_mutual_information.txt'), dtype=np.float64)
    ep_epkl = np.loadtxt(os.path.join(path, 'ep_epkl.txt'), dtype=np.float64)
    ep_mkl = np.loadtxt(os.path.join(path, 'ep_mkl.txt'), dtype=np.float64)

    var = np.loadtxt(os.path.join(path, 'var.txt'), dtype=np.float64)
    varcombo = np.loadtxt(os.path.join(path, 'varcombo.txt'), dtype=np.float64)
    logvar = np.loadtxt(os.path.join(path, 'logvar.txt'), dtype=np.float64)
    logcombo = np.loadtxt(os.path.join(path, 'logcombo.txt'), dtype=np.float64)

    unc_dict = {
        'Total Uncertainty-PE': eoe,
        'Total Uncertainty-EP': ep_eoe,
        'SCR-PE': score,
        'SCR-EP': aep_tu,
        'Data Uncertainty': exe,
        'Mutual Information-PE': mi,
        'Mutual Information-EP': ep_mi,
        'EPKL-PE': epkl,
        'EPKL-EP': ep_epkl,
        'MKL': mkl,
        'ep_MKL': ep_mkl,
        'sMKL-PE': score_npmi,
        'sMKL-EP': npmi,
        'var': var,
        'varcombo': varcombo,
        'logvar': logvar,
        'logcombo': logcombo
    }
    if os.path.exists(os.path.join(path, 'xbleu.txt')):
        xbleu = np.loadtxt(os.path.join(path, 'xbleu.txt'), dtype=np.float64)
        unc_dict['XBLEU'] = xbleu
    if os.path.exists(os.path.join(path, 'xwer.txt')):
        xwer = np.loadtxt(os.path.join(path, 'xwer.txt'), dtype=np.float64)
        unc_dict['XWER'] = xwer

    weights = softmax(lprobs.reshape([-1, beam_width])[:, :n_best] / temp, axis=1)
    assert np.all((np.isfinite(weights)))
    for key in unc_dict.keys():
        uncertainties = unc_dict[key]
        if beam_search:
            unc_dict[key] = np.sum(weights * np.reshape(uncertainties, [-1, beam_width])[:, :n_best], axis=1)
        else:
            unc_dict[key] = np.mean(np.reshape(uncertainties, [-1, beam_width])[:, :n_best], axis=1)
    return unc_dict
