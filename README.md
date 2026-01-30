# Visualization Project
## Visualization, JBI100, Eindhoven University of Technology

### I. What Was Leveraged from Libraries:
General Note: Apart from `dash`, `numpy` and `pandas`, we have utilized all other libraries available in the `requirements.txt`.

1. Standard chart types (scatter, bar, heatmap) via Plotly's API
2. K-means algorithm from scikit-learn (not custom ML implementation)
3. Dash's callback mechanism and component system (though callback logic is custom)
4. Pandas/NumPy for efficient data operations

**In summary**: The visualization designs, interaction patterns, data pipeline, and analytical logic are entirely custom-built, while the project leverages established frameworks (Dash/Plotly) and algorithms (K-means) as building blocks rather than reinventing foundational technologies.

### II. Complexity
- Dual Linking in Tab 1 (Task 1) and Tab 2 (Task 2)
- Linking in Task 3
- Linking the Bubble Chart Updates when filtering by cluster and upon (individual and box) selection of data points with the other diagrams in Tab 4 (Task 4)

### III. How to run this app
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

### IV. Project Structure
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

### V. Task Descriptions
- **Task 1 Capacity and Seasonality Analysis**: Evaluates how bed capacity meets weekly and seasonal patient demand across different services. It utilizes Parallel Coordinates Plots (PCPs) for multivariate analysis of admissions, refusals, and morale, alongside Line Charts to identify temporal trends and the impact of specific events like flu outbreaks.
- **Task 2 Staff Performance and Satisfaction Analysis**: Explores the relationship between staff presence, workload pressure, and patient outcomes. This is implemented via a coordinated four-panel dashboard featuring Scatter Plots to visualize correlations and a Bubble Chart for service-wide benchmarking of efficiency.
- **Task 3 Staff Allocation Timeline**: Aims to optimize staffing levels by providing a temporal comparison of resource supply versus patient demand. It employs a dual-panel layout with Stacked Bar Charts to represent staff roles and a Line Chart for patient admissions, allowing managers to identify cross-departmental demand surges.
- **Task 4 Strategic Operational Clustering**: Identifies structural similarities and operational patterns across departments by applying K-Means clustering to service-week data. It uses a State Space Bubble Chart, a Cluster DNA Heatmap, and a Timeline to detect synchronized "crisis patterns" and analyze system bottlenecks at maximum capacity.
