# Client Segmentation Dashboard

## Project Description

The purpose of this project is to identify and analyze different customer profiles in an e-commerce setting using Data Science tools. The analysis enables the optimization of business strategies based on data, facilitating strategic decision-making through an interactive dashboard developed in Streamlit. The project spans from data preparation and cleaning to visualizing the results through an interactive panel, ensuring that the model and the generated segments are easily interpretable for end-users.

![400077713-8d5cf4e6-7f09-41cb-9a1e-06ac21f585ef](https://github.com/user-attachments/assets/ca22660f-4da5-48d8-8a25-aa95bb8a472d)

---

## Components and Workflow

### **Components**

  ![Database](https://img.shields.io/badge/Database-Data--source-6C757D?style=for-the-badge&logo=data&logoColor=white)
    
-  Contains key information related to customer behavior and characteristics, serving as the primary source of analyzed data.
  
  ![Jupyter](https://img.shields.io/badge/Jupyter_Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
    
-  The notebooks are organized by workflow stages, from initial data cleaning to advanced cluster analysis.
  
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
    
-  A platform used to create the interactive dashboard that visualizes segmentation results and allows comparisons between customer groups.
  
  ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
    
-  A library used for data scaling, clustering algorithm implementation, and evaluation metric computation.

### **Visualization Tools**
  
![Matplotlib](https://img.shields.io/badge/Matplotlib-3766AB?style=for-the-badge&logo=matplotlib&logoColor=white)

- Used to create static plots for exploratory data analysis and to visualize data distributions and trends.

![Seaborn](https://img.shields.io/badge/Seaborn-4C96D7?style=for-the-badge&logoColor=white)

- Enhanced the visualizations with more advanced and aesthetically pleasing statistical plots, such as heatmaps and pairplots.

![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white) 

- Provided interactive visualizations for dynamic data exploration, allowing users to interact with segmentation results directly.

---

## Workflow Summary

![Data Preparation and Cleaning](https://img.shields.io/badge/Data_Preparation_and_Cleaning_(Notebook_0)-0052CC?style=for-the-badge&logo=database&logoColor=white)

- Removal of inconsistent records, including null values, negatives, and zeros in key variables.
   
- Transformation and organization of columns to ensure the quality of the initial data.

---

![Exploratory Analysis](https://img.shields.io/badge/Exploratory_Analysis_(Notebook_1)-228B22?style=for-the-badge&logo=search&logoColor=white)

- Exploration of available variables in the dataset, identifying relevant patterns and trends.
  
- Initial review of possible relationships between main features for customer segmentation.

---

![Clustering Methods and Metrics](https://img.shields.io/badge/Clustering_Methods_and_Metrics_(Notebook_2)-FF4500?style=for-the-badge&logo=analytics&logoColor=white)

- Theoretical evaluation of clustering methods suitable for the segmentation problem.
  
- Selection of clustering approaches based on data nature and definition of metrics to evaluate result quality.

---

![Clustering Implementation](https://img.shields.io/badge/Clustering_Implementation_(Notebook_3)-6A5ACD?style=for-the-badge&logo=chart-bar&logoColor=white)

- Practical application of the methods defined in the theoretical analysis.
  
- Comparison of results to determine the most suitable method based on the behavior of identified groups.

---

![Cluster Analysis and Categorization](https://img.shields.io/badge/Cluster_Analysis_and_Categorization_(Notebook_4)-8A2BE2?style=for-the-badge&logo=pie-chart&logoColor=white)

- Detailed analysis of the generated clusters, identifying specific patterns in customer groups.
  
- Assignment of descriptive labels for each segment based on their main characteristics.

---

![Interactive Dashboard Development](https://img.shields.io/badge/Interactive_Dashboard_Development-20B2AA?style=for-the-badge&logo=dashboard&logoColor=white)

Creation of a dashboard that allows users to:  

- Visualize general statistics.
  
- Explore customer segments.
  
- Compare customer groups interactively.


---

## Technologies Used

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

- Development of the interactive dashboard.
  
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

- Management and analysis of structured data.

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

- Advanced mathematical operations.

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

- Clustering algorithms and evaluation metrics.

![Matplotlib](https://img.shields.io/badge/Matplotlib-3766AB?style=for-the-badge&logo=chart-bar&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4C96D7?style=for-the-badge&logo=area-chart&logoColor=white)

- Static visualizations for exploratory analysis.

![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

- Interactive charts for dynamic analysis.

---

## Setup Instructions

### **Clone the Repository**
```bash
git clone <REPOSITORY_URL>
```

### **Create a Virtual Environment**

Using **venv**:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Using **Conda**:
```bash
conda env create --file environment.yml
conda activate <environment_name>
```

### **Set Up Project Modules in Editable Mode**
To facilitate development, install the modules in editable mode. This ensures that any code changes are reflected immediately.

```bash
cd ./dashboard/
pip install --editable .

cd ../notebooks/
pip install --editable .
```

### **Run the Dashboard**
```bash
streamlit run app.py
```

---

## Installation guide

Please read [install.md](install.md) for details on how to set up this project.

---

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── install.md         <- Detailed instructions to set up this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks                 <- Jupyter notebooks for exploratory and modeling tasks.
    │   │
    │   ├── `0.0_jtd_data_preparation.ipynb`  <- Notebook for loading the raw dataset, cleaning data, 
    │   │ and scaling features. Outputs intermediate datasets saved in the `data/raw/` folder.
    │   │
    │   ├── `1.0_jtd_exploratory_data_analysis.ipynb`   <- Notebook for visualizing data distributions,
    │   │ correlations, and identifying patterns in the dataset (e.g., histograms, pairplots, etc.).
    │   │
    │   ├── `2.0_jtd_selecting_clustering_methods.ipynb`   <- Notebook for experimenting with different 
    │   │ clustering algorithms (e.g., K-means, DBSCAN) and tuning parameters to find the best approach 
    │   │ for the dataset.
    │   │
    │   ├── `3.0_dsb_clustering_implementation.ipynb`   <-  Notebook for applying the selected clustering 
    │   │ method (DBSCAN), training the model, and generating cluster labels for each data point.
    │   │
    │   └── `4.0_dsb_clustering_analysis.ipynb`    <- Notebook for visualizing and analyzing the clustering 
    │     results. Includes scatter plots  (with PCA), box plots, and insights about each cluster's characteristics.
    │
    ├── environment.yml    <- The requirements file for reproducing the analysis environment.
    ├── requirements.txt   <- The pip requirements file for reproducing the environment.
    │
    ├── test               <- Unit and integration tests for the project.
    │   ├── __init__.py
    │   └── test_model.py  <- Example of a test script.
    │
    ├── .here              <- File that will stop the search if none of the other criteria
    │                         apply when searching head of project.
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .)
    │                         so customer_segmentation can be imported.
    │
    └── customer_segmentation   <- Source code for use in this project.
        │
        ├── __init__.py             <- Makes customer_segmentation a Python module.
        │
        ├── config.py               <- Store useful variables and configuration.
        │
        ├── dataset.py              <- Scripts to download or generate data.
        │
        ├── features.py             <- Code to create features for modeling.
        │
        ├── modeling                
        │   ├── __init__.py 
        │   ├── predict.py          <- Code to run model inference with trained models.
        │   └── train.py            <- Code to train models.
        │
        ├── utils                   <- Scripts to help with common tasks.
        │   └── paths.py            <- Helper functions for relative file referencing across the project.        
        │
        └── plots.py                <- Code to create visualizations.

---
Project based on the [cookiecutter conda data science project template](https://github.com/jvelezmagic/cookiecutter-conda-data-science).
