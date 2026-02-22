# src/seeder.py

import numpy as np
import pandas as pd
import random

from src.config import *

def run():
    print('\nLoading model for seed data generation...')

    # Note: This is the order of the columns,
    csv_header_cols = [
        'client_num',
        'attrition_flag',
        'customer_age',
        'gender',
        'dependent_count',
        'education_level',
        'marital_status',
        'income_category',
        'card_category',
        'months_on_book',
        'total_relationship_count',
        'months_inactive_12_mon',
        'contacts_count_12_mon',
        'credit_limit',
        'total_revolving_bal',
        'avg_open_to_buy',
        'total_amt_chng_q4_q1',
        'total_trans_amt',
        'total_trans_ct',
        'total_ct_chng_q4_q1',
        'avg_utilization_ratio',
    ]

    # Create seeder_data.csv
    rng = np.random.default_rng(SEED)

    # Categorical Options
    edu_levels = ['High School', 'Graduate', 'Uneducated', 'College', 'Post-Graduate', 'Doctorate']
    maritals = ['Married', 'Single', 'Divorced']
    incomes = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']
    cards = ['Blue', 'Silver', 'Gold', 'Platinum']

    seeder_data = []

    # Pick how many rows of data that will be created (200 to 10,000 rows).
    num_rows = random.randint(200, 10000)

    for _ in range(num_rows):
        # 1. Determine Attrition (~16% Churn)
        is_churn = rng.random() < CUST_CHURN_RATE
        attrition_flag = 'Attrited Customer' if is_churn else 'Existing Customer'

        # 2. Apply Behavioral Correlation
        if is_churn:
            rel_count = rng.integers(1, 4)      # Lower relationships
            inactive_12 = rng.integers(3, 7)    # Higher inactivity
            trans_ct = rng.integers(10, 50)     # Lower transaction count
            contacts_12 = rng.integers(3, 7)    # Higher bank contacts
        else:
            rel_count = rng.integers(3, 7)      # Higher relationships
            inactive_12 = rng.integers(0, 4)    # Lower inactivity
            trans_ct = rng.integers(40, 140)    # Higher transaction count
            contacts_12 = rng.integers(0, 4)    # Lower bank contacts

        # 3. Generate Numerical values
        client_num = rng.integers(700000000, 899999999)
        age = int(np.clip(rng.normal(46, 8), 26, 73))
        dependents = rng.integers(0, 6)
        months_on_book = int(np.clip(rng.normal(36, 8), 13, 56))

        # Financials
        limit = round(rng.uniform(1438, 34516), 1)
        revolving_bal = rng.integers(0, 2518)

        # Ensure revolving balance doesn't exceed limit
        if revolving_bal > limit:
            revolving_bal = int(limit * 0.5)

        # 4. Enforce Mathematical Logic
        open_to_buy = round(limit - revolving_bal, 1)
        utilization = round(revolving_bal / limit, 3)

        # 5. Other Floats
        amt_chng = round(rng.uniform(0.0, 3.4), 3)
        trans_amt = rng.integers(500, 18501)
        ct_chng = round(rng.uniform(0.0, 3.7), 3)

        seeder_data.append([
            client_num, attrition_flag, age, random.choice(['M', 'F']),
            dependents, random.choice(edu_levels), random.choice(maritals),
            random.choice(incomes), random.choice(cards), months_on_book,
            rel_count, inactive_12, contacts_12, limit, revolving_bal,
            open_to_buy, amt_chng, trans_amt, trans_ct, ct_chng, utilization
        ])

    # Create DataFrame
    output_file = 'data/seeder_data.csv'

    df = pd.DataFrame(seeder_data, columns=csv_header_cols)
    df.to_csv(output_file, index=False)
    print(f"* Successfully generated {num_rows} rows in {output_file}. *")
