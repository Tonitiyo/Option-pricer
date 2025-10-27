import numpy as np

# Variance control 
def mc_call_control_variate(S0, K, r, sigma, T, N):
    Z = np.random.randn(N)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    Y = np.exp(-r*T) * np.maximum(ST - K, 0)
    X = np.exp(-r*T) * ST

    b = np.cov(X, Y)[0,1] / np.var(X)
    Z_corr = Y - b * (X - S0)
    return Z_corr.mean(), Z_corr.std()

print(mc_call_control_variate(40, 40, 0.032, 0.2, 1/3, 100000))

