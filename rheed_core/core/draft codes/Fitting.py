import numpy as np
from scipy.optimize import curve_fit

def normalize_0_1(y, I_start, I_end, I_diff):
    """Normalize y-values to the range [0, 1]."""
    if I_diff is None:
        I_diff = I_end - I_start
    y_nor = (y - I_start) / I_diff
    return y_nor

def de_normalize_0_1(y_nor, I_start, I_end, I_diff):
    """De-normalize y-values from the range [0, 1] back to the original scale."""
    if I_diff is None:
        I_diff = I_end - I_start
    y = y_nor * I_diff + I_start
    return y

def exp_func_inc_simp(x, b1, relax1):
    """Simplified exponential growth function."""
    return b1 * (1 - np.exp(-x / relax1))

def exp_func_dec_simp(x, b2, relax2):
    """Simplified exponential decay function."""
    return b2 * np.exp(-x / relax2)

def exp_func_inc(x, a1, b1, relax1):
    """Full exponential growth function."""
    return (a1 * x + b1) * (1 - np.exp(-x / relax1))

def exp_func_dec(x, a2, b2, relax2):
    """Full exponential decay function."""
    return (a2 * x + b2) * np.exp(-x / relax2)

def normalize_and_extract_amplitude(xs, ys, I_diff, unify):
    ys_nor = []
    I_starts = []
    I_ends = []

    for i in range(len(xs)):
        n_avg = len(ys[i]) // 100 + 3
        I_end = np.mean(ys[i][-n_avg:])
        I_start = np.mean(ys[i][:n_avg])
        I_starts.append(I_start)
        I_ends.append(I_end)
        y_nor = normalize_0_1(ys[i], I_start, I_end, I_diff)
        ys_nor.append(y_nor)

    return ys_nor, I_starts, I_ends

def fit_curves(xs, ys_nor, fit_settings, growth_name):
    parameters = []
    ys_nor_fit, ys_nor_fit_failed, labels, losses = [], [], [], []

    bounds = fit_settings['bounds']
    p_init = fit_settings['p_init']
    unify = fit_settings['unify']

    for i in range(len(xs)):
        x = np.linspace(1e-5, 1, len(ys_nor[i]))  # Use second as x axis unit
        y_nor = ys_nor[i]

        if unify:
            params, _ = curve_fit(exp_func_inc, x, y_nor, p0=p_init, bounds=bounds, absolute_sigma=False)
            a, b, relax = params
            y_nor_fit = exp_func_inc(x, a, b, relax)
            labels.append(f'{growth_name}-index {i + 1}:\ny=({np.round(a, 2)}t+{np.round(b, 2)})*(1-exp(-t/{np.round(relax, 2)}))')
            parameters.append((a, b, relax))
            losses.append((0, 0))
            ys_nor_fit_failed.append(y_nor_fit)
        else:
            params1, _ = curve_fit(exp_func_inc_simp, x, y_nor, p0=p_init[1:], bounds=bounds, absolute_sigma=False)
            b1, relax1 = params1
            y1_nor_fit = exp_func_inc_simp(x, b1, relax1)

            params2, _ = curve_fit(exp_func_dec_simp, x, y_nor, p0=p_init[1:], bounds=bounds, absolute_sigma=False)
            b2, relax2 = params2
            y2_nor_fit = exp_func_dec_simp(x, b2, relax2)

            loss1 = ((y_nor - y1_nor_fit) ** 2).mean()
            loss2 = ((y_nor - y2_nor_fit) ** 2).mean()

            params1_full, _ = curve_fit(exp_func_inc, x, y_nor, p0=p_init, bounds=bounds, absolute_sigma=False)
            a1, b1_full, relax1_full = params1_full
            y1_nor_fit_full = exp_func_inc(x, a1, b1_full, relax1_full)

            params2_full, _ = curve_fit(exp_func_dec, x, y_nor, p0=p_init, bounds=bounds, absolute_sigma=False)
            a2, b2_full, relax2_full = params2_full
            y2_nor_fit_full = exp_func_dec(x, a2, b2_full, relax2_full)

            if loss1 < loss2:
                y_nor_fit = y1_nor_fit_full
                labels.append(f'{growth_name}-index {i + 1}:\ny1=({np.round(a1, 2)}t+{np.round(b1_full, 2)})*(1-exp(-t/{np.round(relax1_full, 2)}))')
                parameters.append((a1, b1_full, relax1_full))
                ys_nor_fit_failed.append(y2_nor_fit_full)
            else:
                y_nor_fit = y2_nor_fit_full
                labels.append(f'{growth_name}-index {i + 1}:\ny2=({np.round(a2, 2)}t+{np.round(b2_full, 2)})*(exp(-t/{np.round(relax2_full, 2)}))')
                parameters.append((a2, b2_full, relax2_full))
                ys_nor_fit_failed.append(y1_nor_fit_full)

            losses.append((loss1, loss2))

        ys_nor_fit.append(y_nor_fit)

    return parameters, ys_nor_fit, ys_nor_fit_failed, labels, losses

def de_normalize_and_assemble(xs, ys, ys_nor_fit, I_starts, I_ends, I_diff, unify):
    ys_fit = []

    for i in range(len(xs)):
        y_fit = de_normalize_0_1(ys_nor_fit[i], I_starts[i], I_ends[i], I_diff)
        ys_fit.append(y_fit)

    return ys_fit

def fit_exp_function(xs, ys, growth_name, fit_settings={'I_diff': None, 'unify': True, 'bounds': [0.01, 1], 'p_init': (1, 0.1, 0.4), 'n_std': 1}):
    """
    Fits an exponential function to the given data.

    Args:
        xs (list of list of floats): List of x-values for each partial curve. Each element is a list of x-values for one curve.
        ys (list of list of floats): List of y-values for each partial curve. Each element is a list of y-values for one curve.
        growth_name (str): Name of the growth process, used for labeling the fitted curves.
        fit_settings (dict, optional): Dictionary of settings for the fitting process. Defaults to:
            {
                'I_diff': None,          # Optional intensity difference for normalization.
                'unify': True,           # Whether to use a unified fitting function for all curves.
                'bounds': [0.01, 1],     # Bounds for the fitting parameters.
                'p_init': (1, 0.1, 0.4), # Initial guess for the fitting parameters.
                'n_std': 1               # Number of standard deviations for normalization.
            }

    Returns:
        tuple:
            - numpy.ndarray: Array of fitted parameters for each curve.
            - list: A list containing:
                - xs: Original x-values.
                - ys: Original y-values.
                - ys_fit: Fitted y-values.
                - ys_nor: Normalized y-values.
                - ys_nor_fit: Fitted normalized y-values.
                - ys_nor_fit_failed: Failed fitted normalized y-values (when applicable).
                - labels: Labels for each fitted curve.
                - losses: Losses for each fitted curve.
    """
    I_diff = fit_settings['I_diff']
    unify = fit_settings['unify']

    # Step 1: Normalize and extract amplitude
    ys_nor, I_starts, I_ends = normalize_and_extract_amplitude(xs, ys, I_diff, unify)

    # Step 2: Fit curves
    parameters, ys_nor_fit, ys_nor_fit_failed, labels, losses = fit_curves(xs, ys_nor, fit_settings, growth_name)

    # Step 3: De-normalize and assemble results
    ys_fit = de_normalize_and_assemble(xs, ys, ys_nor_fit, I_starts, I_ends, I_diff, unify)

    return np.array(parameters), [xs, ys, ys_fit, ys_nor, ys_nor_fit, ys_nor_fit_failed, labels, losses]
