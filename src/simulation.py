import numpy as np

def monte_carlo(F, V, mu, lambda_, E_C, n=1000):

    results = []

    for _ in range(n):
        I_t = np.random.normal(mu, 0.01)
        N_t = np.random.poisson(lambda_)
        costs = np.random.randint(300, 800, N_t)

        X_t = F + V*(1 + I_t) + sum(costs)
        results.append(X_t)

    return results


# NEW advanced function
def advanced_simulation(F, V, mu, months=12, income=5000):

    results = []
    savings = []

    current_expense = V
    current_income = income

    for month in range(months):

        inflation = np.random.normal(mu, 0.01)
        current_expense = current_expense * (1 + inflation)

        if month % 6 == 0 and month != 0:
            current_income *= 1.05

        event_cost = 0

        if np.random.rand() < 0.3:
            event_cost += np.random.randint(500, 2000)

        if np.random.rand() < 0.2:
            event_cost += np.random.randint(1000, 5000)

        if np.random.rand() < 0.25:
            event_cost += np.random.randint(300, 1500)

        if np.random.rand() < 0.4:
            event_cost += np.random.randint(200, 1000)

        if month in [10, 11]:
            event_cost += np.random.randint(2000, 5000)

        total_expense = F + current_expense + event_cost
        saving = current_income - total_expense

        results.append(total_expense)
        savings.append(saving)

    return results, savings