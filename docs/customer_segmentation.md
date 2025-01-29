# **Customer Segmentation Clustering - Documentation**

## **Overview**
This document describes how the customer segmentation solution works, including two main components: **the automated clustering pipeline** and **the interactive dashboard**. These tools allow businesses to analyze customer behavior, create targeted strategies, and optimize decision-making.

---

## **Solution Workflow**

### ![Automated Clustering Pipeline](https://img.shields.io/badge/Automated_Clustering_Pipeline-20B2AA?style=for-the-badge)

The **clustering pipeline** automates customer segmentation using **DBSCAN** clustering. It processes a **pre-cleaned dataset** and assigns each record to a cluster based on density estimation.

#### **Workflow:**
- **Data Input:**
  - Requires a **preprocessed dataset** (cleaned and structured) stored in `data/processed/`.
  - The dataset must contain numerical features relevant to customer segmentation.
  - It should match the structure of the original dataset to ensure compatibility.

- **Execution:**
  - The pipeline scales the dataset using `MinMaxScaler`.
  - **DBSCAN clustering** is applied to detect customer groups.
  - Outputs a **dataset with assigned cluster labels**.

- **Output:**
  - A dataset with an added `cluster` column.
  - A printed summary of cluster distributions.

#### **Customization:**
- Modify the dataset path in `customer_segmentation/clustering.py`:
  ```python
  dataset_path = data_processed_dir("your_dataset.csv")
  ```
- Adjust clustering parameters in `create_pipeline()`:
  ```python
  def create_pipeline():
      return DBSCANPipeline([
          ('scaler', MinMaxScaler()),
          ('dbscan', DBSCAN(eps=0.04, min_samples=50))
      ])
  ```
- Ensure that any new dataset maintains **similar feature distributions** for accurate clustering.

### **Why is this useful?**
- **Identifies customer groups** based on behavioral patterns.
- **Optimizes marketing strategies** by targeting specific segments.
- **Improves business decisions** by understanding different customer needs.

---

### ![Interactive Dashboard](https://img.shields.io/badge/Interactive_Dashboard-FF4B4B?style=for-the-badge)

The **dashboard** provides an interactive interface for visualizing customer segmentation.

#### **Workflow:**
- **Data Input:**
  - Loads a **clean dataset** (not yet clustered) from `data/processed/`.
  - Applies **DBSCAN clustering dynamically** at runtime.
  - Requires numerical features structured similarly to the original dataset.

- **Execution:**
  - Loads and preprocesses the dataset.
  - Runs **DBSCAN clustering** dynamically.
  - Displays results in an interactive format.

- **Features:**
  - **Overview Section:** Summarizes dataset insights.
  - **Cluster Analysis Section:** Visualizes customer groups.

#### **Customization:**
- Update the dataset path in `dashboard/data.py`:
  ```python
  dataset_path = data_processed_dir("your_cleaned_dataset.csv")
  ```
- Modify clustering parameters in `load_data()` if necessary:
  ```python
  dbscan = DBSCAN(eps=0.040, min_samples=50)
  ```
- Ensure the dataset contains **relevant customer attributes** to maintain clustering quality.

### **Why is this useful?**
- **Facilitates data exploration** through an interactive UI.
- **Enhances customer segmentation visibility** with dynamic graphs.
- **Supports business decisions** by providing data-driven insights.

---

## **Execution**

To run the solution:
- **Clustering Pipeline:** Execute `clustering.py` in the `customer_segmentation/` folder.
- **Dashboard:** Run `app.py` in the `dashboard/` folder.

To use a different dataset, update the relevant file paths in `clustering.py` (for segmentation) and `data.py` (for the dashboard). The dataset must be structured similarly to the original data for compatibility.
