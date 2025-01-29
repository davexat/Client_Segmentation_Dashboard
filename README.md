# **Client Segmentation Dashboard**

## **Project Overview**

The main objective of this project is to identify and analyze different customer profiles in an e-commerce environment using advanced Data Science techniques. This segmentation enables businesses to optimize their strategies by understanding customer behavior, driving informed decision-making processes. An interactive dashboard serves as the final deliverable, offering stakeholders an intuitive platform to explore segmentation results. This project encompasses the entire data pipeline, from raw data preprocessing to customer clustering and visualization, ensuring actionable insights are accessible even to non-technical users. Additionally, an automated pipeline was developed with the best clustering model to scale the segmentation process for new datasets.

![400077713-8d5cf4e6-7f09-41cb-9a1e-06ac21f585ef](https://github.com/user-attachments/assets/ca22660f-4da5-48d8-8a25-aa95bb8a472d)

---

## **Components and Workflow**

### **Components**

#### ![Data Source](https://img.shields.io/badge/Data_Source-6C757D?style=for-the-badge)  

- Serves as the foundation, containing key transactional and behavioral data used for customer segmentation.  
- Data is structured to ensure reliability during preprocessing and analysis phases.

#### ![Notebooks](https://img.shields.io/badge/Notebooks-F37626?style=for-the-badge)  

Used during iterative development and experimentation:  

- **Notebook 0:** Data preparation and cleaning.
- **Notebook 1:** Exploratory data analysis (EDA) to identify trends and patterns.  
- **Notebook 2:** Selection of clustering methods based on metrics and theoretical evaluation.  
- **Notebook 3:** Implementation of clustering models and result evaluation.  
- **Notebook 4:** Final analysis and labeling of clusters with descriptive summaries.

#### ![Machine Learning and Preprocessing](https://img.shields.io/badge/Machine_Learning_and_Preprocessing-F7931E?style=for-the-badge)  

- Data scaling and transformation to optimize clustering algorithms.  
- Implementation and evaluation of clustering techniques using model performance metrics such as silhouette scores.  

#### ![Visualization Tools](https://img.shields.io/badge/Visualization_Tools-3F4F75?style=for-the-badge)  

- Provides in-depth exploratory analysis through static and interactive visualizations.  
- Integrated with the dashboard for dynamic user interaction.  

#### ![Automated Pipeline](https://img.shields.io/badge/Automated_Pipeline-20B2AA?style=for-the-badge)  

- A standalone pipeline was developed using the best clustering model, enabling automated customer segmentation for new datasets.  
- Ensures reproducibility and scalability for long-term business applications.  

#### ![Interactive Dashboard](https://img.shields.io/badge/Interactive_Dashboard-FF4B4B?style=for-the-badge)  

- Allows real-time visualization and exploration of customer segmentation.  
- Provides comparative analysis across clusters for business decision-making.  

---

## **Workflow Summary**

### **1. Data Preparation and Cleaning**  

- Removal of duplicate or inconsistent records.  
- Standardization of numerical variables to improve clustering accuracy.  
- Transformation of categorical variables into numerical formats where applicable.  

### **2. Exploratory Data Analysis (EDA)**  

- Identification of key behavioral patterns in the dataset.  
- Visualization of statistical distributions and feature correlations.  

### **3. Clustering Method Selection**  

- Testing multiple clustering algorithms and assessing performance with silhouette and Davies-Bouldin scores.  
- Selection of the optimal clustering model based on data characteristics.  

### **4. Clustering Implementation**  

- Application of the chosen clustering model to segment customers.  
- Evaluation of cluster cohesiveness and separation through statistical and visual methods.  

### **5. Cluster Analysis and Labeling**  

- In-depth analysis of each cluster’s distinguishing traits.  
- Assignment of meaningful labels for enhanced interpretability.  

### **6. Dashboard Development**  

The dashboard enables stakeholders to:  

- Analyze general statistics and key segmentation metrics.  
- Interact dynamically with customer clusters.  
- Compare different groups through an intuitive UI.  

---

## **Tech Stack**

### **Programming Language**  

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  

  The core language used for all project components.

### **Development and Experimentation**  

![Jupyter](https://img.shields.io/badge/Jupyter_Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)  

  Environment for iterative data processing, visualization, and model development.

### **Data Handling and Analysis**  

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
  
  Data manipulation and preprocessing.  
  
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
    
  Advanced numerical computation and matrix operations.  

### **Machine Learning**  

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
  
  Clustering algorithms, preprocessing, and performance evaluation.  

### **Visualization Tools**  

![Matplotlib](https://img.shields.io/badge/Matplotlib-3766AB?style=for-the-badge&logo=matplotlib&logoColor=white)![Seaborn](https://img.shields.io/badge/Seaborn-4C96D7?style=for-the-badge)

  For static and statistical visualizations.

![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)  

  For interactive and dynamic data visualizations.  

### **Dashboard Development**  

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)  

  Framework for building the interactive segmentation dashboard.

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
