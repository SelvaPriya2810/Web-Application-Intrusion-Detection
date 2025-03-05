

# **Web Application Intrusion Detection Using Machine Learning**  

## **Overview**  
This project focuses on detecting web application intrusions using **machine learning techniques**. The dataset used is the **KDD Cup 99 dataset**, which contains network traffic data labeled as normal or attack types. The goal is to classify network requests into different categories to enhance **cybersecurity**.

## **Project Features**  
- **Data Preprocessing**: Handling missing values, encoding categorical features, feature scaling, and dimensionality reduction.  
- **Exploratory Data Analysis (EDA)**: Visualizing dataset distribution, attack types, and correlation between features.  
- **Feature Engineering**: Identifying important network parameters that indicate potential intrusions.  
- **Machine Learning Models**:  
  - **Decision Tree Classifier (DTC)**  
  - **Random Forest Classifier (RF)**  
  - **Support Vector Classifier (SVC)**  
  - **Logistic Regression (LR)**  
  - **Gaussian Naive Bayes (GNB)**  
- **Evaluation Metrics**:  
  - Accuracy, Precision, Recall, F1-score.  
  - Confusion Matrix and ROC Curves.  
- **Comparative Analysis**: Evaluating different models based on accuracy and execution time.  

## **Dataset**  
- **KDD Cup 99 Dataset**  
- **Total Samples**: 494,021  
- **Features**: 42 (after preprocessing: 32 features)  
- **Classes**:  
  - **Normal**  
  - **DoS (Denial of Service Attacks)**  
  - **Probe (Surveillance and Probing Attacks)**  
  - **R2L (Remote to Local Attacks)**  
  - **U2R (User to Root Attacks)**  

## **Technologies Used**  
- **Python**: Programming language.  
- **Pandas & NumPy**: Data manipulation and analysis.  
- **Matplotlib & Seaborn**: Data visualization.  
- **Scikit-learn**: Machine learning models and evaluation.  

## **Installation**  

### **Prerequisites**  
Ensure you have the following installed:  
- Python 3.x  
- Required libraries (install using pip)  

```bash
pip install pandas numpy matplotlib seaborn scikit-learn missingno
```

### **Running the Project**  
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/SelvaPriya2810/Web-Application-Intrusion-Detection.git
   ```
2. **Navigate to the project folder**:  
   ```bash
   cd Web-Application-Intrusion-Detection
   ```
3. **Run the Jupyter Notebook** for model training and evaluation.  
   ```bash
   jupyter notebook projectFinal.ipynb
   ```
4. **Analyze results and compare different models based on accuracy and computation time.**  

## **Results & Observations**  
- **Random Forest and Support Vector Classifier achieved the highest accuracy** (~99.9%).  
- **Gaussian Naive Bayes had the lowest accuracy**, performing well on DoS attacks but struggling with R2L and U2R attacks.  
- **Support Vector Classifier had the longest training time**, making it computationally expensive.  
- **Decision Tree and Logistic Regression provided good accuracy with lower computational costs.**  

