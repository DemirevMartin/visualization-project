COLOR_SUFFICIENT = "#1a73b2"
COLOR_SHORTAGE = "#d62728"

COLOR_PATIENTS_ADMITTED = "#5fa3d9"
COLOR_PATIENT_SATISFACTION = "#e8625f"
COLOR_STAFF_MORALE = "#5dc86a"
COLOR_PATIENTS_REFUSED = "#9e74cc"

COLOR_EMERGENCY = "#d62728"
COLOR_ICU = "#ff7f0e"
COLOR_SURGERY = "#2ca02c"
COLOR_GENERAL_MEDICINE = "#1f77b4"

COLOR_DOCTOR = "#D5E925"
COLOR_NURSE = "#FF0000"
COLOR_NURSING_ASSISTANT = "#0004FF"

# Cluster Colors (Low to High Stress)
COLOR_CLUSTER_LOW = "#52c652"
COLOR_CLUSTER_MED_LOW = "#3f7ae0"
COLOR_CLUSTER_MED_HIGH = "#F32ADB"
COLOR_CLUSTER_HIGH = "#d54848"
COLOR_CLUSTER_VERY_HIGH = "#692baa"

COLORS_DICT = {
    # PCP Colors
    'sufficient': COLOR_SUFFICIENT,
    'shortage': COLOR_SHORTAGE,

    # data columns colors
    'patients_admitted': COLOR_PATIENTS_ADMITTED,
    'patient_satisfaction': COLOR_PATIENT_SATISFACTION,
    'staff_morale': COLOR_STAFF_MORALE,
    'patients_refused': COLOR_PATIENTS_REFUSED,

    # Service Colors
    'emergency': COLOR_EMERGENCY,
    'ICU': COLOR_ICU,
    'surgery': COLOR_SURGERY,
    'general_medicine': COLOR_GENERAL_MEDICINE,

    # Role Colors
    'doctor': COLOR_DOCTOR,
    'nurse': COLOR_NURSE,
    'nursing_assistant': COLOR_NURSING_ASSISTANT,

    # Cluster Colors
    'cluster_low': COLOR_CLUSTER_LOW,
    'cluster_med_low': COLOR_CLUSTER_MED_LOW,
    'cluster_med_high': COLOR_CLUSTER_MED_HIGH,
    'cluster_high': COLOR_CLUSTER_HIGH,
    'cluster_very_high': COLOR_CLUSTER_VERY_HIGH
}

# To make it easier for the cluster coloring
CLUSTER_COLORS = [
    COLOR_CLUSTER_LOW,
    COLOR_CLUSTER_MED_LOW,
    COLOR_CLUSTER_MED_HIGH,
    COLOR_CLUSTER_HIGH,
    COLOR_CLUSTER_VERY_HIGH
]
