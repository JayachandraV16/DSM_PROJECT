import numpy as np

def preprocess(df):

    df['total_expense'] = df[
        ['Rent','Loan_Repayment','Insurance','Groceries','Transport',
         'Eating_Out','Entertainment','Utilities','Healthcare','Education','Miscellaneous']
    ].sum(axis=1)

    df['risk_category'] = np.where(df['total_expense'] > df['Income']*0.8, 1, 0)

    # Add stochastic variables
    df['inflation_rate'] = np.random.normal(0.05, 0.01, len(df))
    df['unexpected_events'] = np.random.poisson(1.5, len(df))
    df['unexpected_cost'] = np.random.randint(300, 800, len(df))

    return df