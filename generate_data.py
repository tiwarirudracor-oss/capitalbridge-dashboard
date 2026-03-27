import pandas as pd
import numpy as np

def generate_cb_data():
    np.random.seed(42)
    n = 2000
    
    # Core Demographics
    regions = ['India', 'UAE']
    industries = ['Tech/SaaS', 'Manufacturing', 'Retail', 'Healthcare', 'Real Estate', 'Logistics']
    challenges = ['Regulatory Compliance', 'Investor Access', 'Cash Flow Management', 'Valuation Gaps']
    
    data = {
        'Company_ID': [f'CB_{i:04d}' for i in range(1, n + 1)],
        'Region': np.random.choice(regions, n, p=[0.6, 0.4]),
        'Industry': np.random.choice(industries, n),
        'Revenue_USD': np.random.lognormal(mean=14.5, sigma=1.0, size=n),
        'Growth_Rate': np.random.uniform(5, 120, n),
        'Audit_Readiness': np.random.randint(1, 11, n),
        'Client_Concentration': np.random.uniform(10, 85, n),
        'Independent_Board': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'Challenge': np.random.choice(challenges, n)
    }
    
    df = pd.DataFrame(data)
    
    # Engineering dependent variables for ML logic
    # Funding needed is roughly 1.5x - 3x Revenue for high growth firms
    df['Funding_Needed_USD'] = df['Revenue_USD'] * (df['Growth_Rate']/100) * np.random.uniform(1.2, 2.5, n)
    
    # Classification Target: Propensity to Engage (0: Low, 1: High)
    # High if growth is high, audit readiness is low (needs help), or facing 'Investor Access' challenges
    engage_prob = (df['Growth_Rate'] * 0.4) + ((10 - df['Audit_Readiness']) * 5) + (df['Challenge'] == 'Investor Access').astype(int) * 20
    df['Propensity_Label'] = (engage_prob > engage_prob.median()).astype(int)
    
    df.to_csv('capitalbridge_data.csv', index=False)
    print("Dataset 'capitalbridge_data.csv' generated with 2000 records.")

if __name__ == "__main__":
    generate_cb_data()
