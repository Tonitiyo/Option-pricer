#Test 

import pandas as pd
from pricing import monte_carlo_call_put

def convergence_test(S_0, r, vol, T, K):
    results = []
    for n in [10, 100, 1000, 10000, 100000]:
        call, put = monte_carlo_call_put(S_0, r, vol, T, K, n)
        results.append({'Simulations': n, 'Call': call, 'Put': put})
    return pd.DataFrame(results)

print(convergence_test(40, 0.03, 0.2, 1/3, 40))