COLOR_SUFFICIENT = "#1a73b2"
COLOR_SHORTAGE = "#d62728"

COLOR_PATIENTS_ADMITTED = "#1a73b2"
COLOR_PATIENT_SATISFACTION = "#9e74cc"
COLOR_STAFF_MORALE = "#049414"
COLOR_PATIENTS_REFUSED = "#d62728"

COLOR_EMERGENCY = "#e0770d"
COLOR_ICU = "#07dbf7"
COLOR_SURGERY = "#f1f707"
COLOR_GENERAL_MEDICINE = "#1fd368"

COLOR_DOCTOR = "#9e74cc"
COLOR_NURSE = "#049414"
COLOR_NURSING_ASSISTANT = "#0605ad"

# Cluster Colors (Low to High Stress)
COLOR_CLUSTER_LOW = "#fde0e0"
COLOR_CLUSTER_MED_LOW = "#f4a6a6"
COLOR_CLUSTER_MED_HIGH = "#d62728"
COLOR_CLUSTER_HIGH = "#a51d1d"
COLOR_CLUSTER_VERY_HIGH = "#6f1212"

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
