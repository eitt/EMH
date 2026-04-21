import numpy as np
import scipy.stats as stats

def diebold_mariano_test(y_true, y_pred_baseline, y_pred_model, h=1, loss='mse'):
    """
    Computes the Diebold-Mariano test statistic and p-value.
    Null hypothesis: Predictive accuracy of the baseline and model are equal.
    Alternative hypothesis: Predictive accuracy of the model is different from the baseline.
    
    Args:
        y_true: True target values.
        y_pred_baseline: Predictions from the baseline model.
        y_pred_model: Predictions from the proposed model (diffusion).
        h: Forecast horizon (used for autocovariance correction).
        loss: 'mse' (Mean Squared Error) or 'mae' (Mean Absolute Error).
        
    Returns:
        DM statistic, p-value.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_baseline = np.asarray(y_pred_baseline).flatten()
    y_pred_model = np.asarray(y_pred_model).flatten()
    
    if len(y_true) != len(y_pred_baseline) or len(y_true) != len(y_pred_model):
        raise ValueError("Inputs have different lengths")

    if loss == 'mse':
        e_baseline = (y_true - y_pred_baseline) ** 2
        e_model = (y_true - y_pred_model) ** 2
    elif loss == 'mae':
        e_baseline = np.abs(y_true - y_pred_baseline)
        e_model = np.abs(y_true - y_pred_model)
    else:
        raise ValueError("Invalid loss type")
        
    # Differential loss: positive means the model is better (smaller error)
    d = e_baseline - e_model
    d_mean = np.mean(d)
    
    T = float(len(d))
    gamma0 = np.var(d, ddof=0)
    
    # Estimate autocovariance of difference up to lag h-1
    gamma = np.zeros(h)
    for i in range(1, h):
        if T - i > 0:
            gamma[i] = np.mean((d[i:] - d_mean) * (d[:-i] - d_mean))
            
    v_d = gamma0 + 2 * np.sum(gamma)
    
    if v_d <= 0:
        return 0, 1.0  # Cannot compute variance properly

    # Diebold-Mariano statistic
    dm_stat = d_mean / np.sqrt(v_d / T)
    
    # p-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

