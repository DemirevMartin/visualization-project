# Visualization Project
### Visualization, JBI100, Eindhoven University of Technology


## I. How to run this app
1. Create a virtual environment
```
> python -m venv .venv
```
If python is not recognized, use the keyword `python3` instead.

2. Activate the virtual environment
- In Windows: 
```
> .\.venv\Scripts\activate
```

- In Unix system:
```
> source .venv/bin/activate
```

3. Install all required packages
```
> pip install -r requirements.txt
```
4. To run the dashboard
```
> cd dashboard
> python app.py
```

## II. Project Structure
1. `/dashboard`
    - `app.py`: is the main file that handles the app
    - `loader.py`: loads and preprocesses the data and what is globally needed
    - `colors.py`: define global colors to ensure consistency across the app
    - `/diagrams`
        - `tab1.py`: contains the implementation of Task 1
        - `tab2.py`: contains the implementation of Task 2 and Task 3
        - `tab3.py`: contains the implementation of Task 4
2. `/exploration`
    - `exploration.ipynb`: contains the initial exploration and general quality checks of the data
3. `/data`: contains the datasets in csv format and a readme with a general explanation


## III. Task Descriptions
- **Task 1 Capacity and Seasonality Analysis**: Evaluates how bed capacity meets weekly and seasonal patient demand across different services. It utilizes Parallel Coordinates Plots (PCPs) for multivariate analysis of admissions, refusals, and morale, alongside Line Charts to identify temporal trends and the impact of specific events like flu outbreaks.
- **Task 2 Staff Performance and Satisfaction Analysis**: Explores the relationship between staff presence, workload pressure, and patient outcomes. This is implemented via a coordinated four-panel dashboard featuring Scatter Plots to visualize correlations and a Bubble Chart for service-wide benchmarking of efficiency.
- **Task 3 Staff Allocation Timeline**: Aims to optimize staffing levels by providing a temporal comparison of resource supply versus patient demand. It employs a dual-panel layout with Stacked Bar Charts to represent staff roles and a Line Chart for patient admissions, allowing managers to identify cross-departmental demand surges.
- **Task 4 Strategic Operational Clustering**: Identifies structural similarities and operational patterns across departments by applying K-Means clustering to service-week data. It uses a State Space Bubble Chart, a Cluster DNA Heatmap, and a Timeline to detect synchronized "crisis patterns" and analyze system bottlenecks at maximum capacity.
