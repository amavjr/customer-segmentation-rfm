import pandas as pd
import numpy as np

# Create sample transaction data for RFM analysis
np.random.seed(42)
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
customers = range(1, 101)

data = []
for customer in customers:
    n_transactions = np.random.randint(1, 20)
    transaction_dates = np.random.choice(dates, n_transactions, replace=False)
    for date in transaction_dates:
        amount = np.random.uniform(10, 500)
        data.append([customer, date, amount])

df = pd.DataFrame(data, columns=['CustomerID', 'InvoiceDate', 'Amount'])

# Calculate RFM metrics
snapshot_date = pd.Timestamp('2024-12-31')
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'CustomerID': 'count',
    'Amount': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'CustomerID': 'Frequency',
    'Amount': 'Monetary'
})

# Create RFM scores
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Segment customers
def segment_customers(row):
    if row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
        return 'Champions'
    elif row['R_Score'] >= 3 and row['F_Score'] >= 2:
        return 'Loyal'
    elif row['R_Score'] >= 4 and row['F_Score'] == 1:
        return 'New Customers'
    elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
        return 'At Risk'
    elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
        return 'Lost'
    else:
        return 'Needs Attention'

rfm['Segment'] = rfm.apply(segment_customers, axis=1)

print("\n📊 RFM Segmentation Summary:")
print(rfm['Segment'].value_counts())

rfm.to_csv('data/rfm_segments.csv')
print("\n✅ RFM segments saved to 'data/rfm_segments.csv'")
