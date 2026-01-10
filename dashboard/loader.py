import pandas as pd
import numpy as np

# Adjust paths if necessary
SERVICES_PATH = "../data/services_weekly.csv"
SCHEDULE_PATH = "../data/staff_schedule.csv"
PATIENTS_PATH = "../data/patients.csv"
STAFF_PATH = "../data/staff.csv"

def load_data():
    """
    Loads, merges, and preprocesses all datasets.
    Returns a single 'rich' DataFrame with all metrics for all tasks.
    """
    # ---------------------------
    # 1. Load Services (Base DF)
    # ---------------------------
    df = pd.read_csv(SERVICES_PATH)
    
    # ---------------------------
    # 2. Load Patients & Aggregate
    # ---------------------------
    patients = pd.read_csv(PATIENTS_PATH)
    patients['arrival_date'] = pd.to_datetime(patients['arrival_date'])
    patients['departure_date'] = pd.to_datetime(patients['departure_date'])
    patients['length_of_stay'] = (patients['departure_date'] - patients['arrival_date']).dt.days
    patients['week'] = patients['arrival_date'].dt.isocalendar().week
    
    # Aggregate patient details by week/service
    patient_agg = patients.groupby(['week', 'service']).agg({
        'satisfaction': ['mean', 'std', 'min', 'max'],
        'length_of_stay': ['mean', 'median', 'std', 'max']
    }).reset_index()
    
    # Flatten columns
    patient_agg.columns = [
        'week', 'service', 
        'avg_satisfaction', 'std_satisfaction', 'min_satisfaction', 'max_satisfaction',
        'avg_los', 'median_los', 'std_los', 'max_los'
    ]
    
    # Merge into main DF
    df = pd.merge(df, patient_agg, on=['week', 'service'], how='left')

    # ---------------------------
    # 3. Load Staff Schedule
    # ---------------------------
    schedule = pd.read_csv(SCHEDULE_PATH)
    
    # Calculate Total vs Present
    staff_metrics = schedule.groupby(['week', 'service']).agg({
        'present': ['sum', 'count']
    }).reset_index()
    staff_metrics.columns = ['week', 'service', 'staff_present', 'total_staff']
    
    # Calculate Presence Rate
    staff_metrics['presence_rate'] = staff_metrics['staff_present'] / staff_metrics['total_staff']
    
    # Merge into main DF
    df = pd.merge(df, staff_metrics, on=['week', 'service'], how='left')

    # Pivot Roles (for Stacked Bars in Task 4 & Correlation in Task 3)
    # We filter only 'present' staff for the count
    roles = schedule[schedule['present'] == 1].pivot_table(
        index=['week', 'service'], 
        columns='role', 
        values='staff_id', 
        aggfunc='count', 
        fill_value=0
    ).reset_index()
    
    # Ensure columns exist even if data is missing
    for r in ['doctor', 'nurse', 'nursing_assistant']:
        if r not in roles.columns:
            roles[r] = 0
            
    df = pd.merge(df, roles, on=['week', 'service'], how='left')

    # ---------------------------
    # 4. Feature Engineering
    # ---------------------------
    df.fillna(0, inplace=True)

    # Task 1 & 5: Availability & Occupancy
    df['availability_status'] = df.apply(lambda x: 'Shortage' if x['patients_request'] > x['available_beds'] else 'Sufficient', axis=1)
    df['status_id'] = df['availability_status'].map({'Sufficient': 0, 'Shortage': 1})
    df['occupancy_rate'] = df.apply(lambda x: (x['patients_admitted'] / x['available_beds']) * 100 if x['available_beds'] > 0 else 0, axis=1)

    # Task 3: Complex Staffing Metrics
    df['patients_per_staff'] = df['patients_admitted'] / df['staff_present'].replace(0, np.nan)
    df['workload'] = df['patients_request'] / df['staff_present'].replace(0, np.nan)
    df['understaffed'] = df['presence_rate'] < 0.75
    
    median_req = df['patients_request'].median()
    df['busy_week'] = df['patients_request'] > median_req

    # Presence Categories (for Box Plots/Bar Charts)
    df['presence_category'] = pd.cut(
        df['presence_rate'],
        bins=[-0.1, 0.6, 0.75, 0.9, 1.1],
        labels=['Very Low (<60%)', 'Low (60-75%)', 'Medium (75-90%)', 'High (90%+)']
    )

    # Combined Condition
    def get_condition(row):
        if row['busy_week'] and row['understaffed']: return 'Busy & Understaffed'
        if row['busy_week'] and not row['understaffed']: return 'Busy & Well Staffed'
        if not row['busy_week'] and row['understaffed']: return 'Not Busy & Understaffed'
        return 'Not Busy & Well Staffed'
    df['condition'] = df.apply(get_condition, axis=1)

    return df