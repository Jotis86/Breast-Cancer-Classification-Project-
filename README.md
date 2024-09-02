# 🎀 Breast Cancer Classification Project 🎀

## 🎯 Objective
- The objective of this project is to develop and optimize machine learning models to classify breast cancer tumors as malignant or benign using the load_breast_cancer dataset from scikit-learn. The goal is to achieve high accuracy and reliability in predictions to aid in medical diagnostics.

## ⚙️ Functionality
- This project involves:
  - 📥 Loading and preprocessing the breast cancer dataset.
  - 🧠 Training and evaluating initial machine learning models (Random Forest and K-Nearest Neighbors).
  - 🔧 Optimizing the models using GridSearchCV to find the best hyperparameters.
  - 📊 Comparing the performance of the initial and optimized models.
  - 🎨 Visualizing the results to understand the model performance better.

## 🛠️ Tools Used
- Python 🐍: Programming language used for the project.
- scikit-learn 📚: Machine learning library for model building and evaluation.
- pandas 🐼: Data manipulation and analysis.
- numpy 🔢: Numerical computing.
- matplotlib 📈: Plotting and visualization.
- seaborn 🌊: Statistical data visualization.

## 🛤️ Development Process
- Data Loading and Preprocessing
  - Loading the Dataset: The load_breast_cancer dataset from scikit-learn was loaded into a pandas DataFrame for easier manipulation and analysis.
  - Inspecting the Dataset: Checked for missing values and obtained descriptive statistics to understand the data distribution and identify any potential issues.
  - Standardizing the Features: Standardized the features to ensure they are on the same scale, which is crucial for many machine learning algorithms.
- Initial Model Training
  - Random Forest Classifier: Trained a Random Forest classifier using the default parameters.
  - K-Nearest Neighbors (KNN) Classifier: Trained a KNN classifier using the default parameters.
  - Evaluation: Evaluated both models using accuracy, precision, recall, and F1-score to understand their initial performance.
- Model Optimization
  - Hyperparameter Tuning with GridSearchCV: Used GridSearchCV to find the best hyperparameters for both the Random Forest and KNN models.
  - Random Forest: Tuned parameters such as n_estimators, max_features, max_depth, and criterion.
  - KNN: Tuned parameters such as n_neighbors, weights, and metric.
  - Evaluation of Optimized Models: Evaluated the optimized models using the same metrics to compare their performance with the initial models.
- Visualization
  - Class Distribution: Created a bar plot to show the distribution of malignant and benign cases in the dataset.
  - Correlation Matrix: Generated a heatmap to display the correlation between different features in the dataset.
  - Feature Distributions: Plotted histograms to show the distribution of key features like mean radius and mean texture.
  - Boxplots by Class: Created boxplots to compare the distribution of features like mean radius and mean texture between malignant and benign classes.
  - Pairplot of Selected Features: Generated a pairplot to visualize the relationships between selected features, colored by class.

## 🏆 Results
- Initial Models
- Random Forest 🌳:
  - Accuracy: 0.96 🎯
  - Precision (Malignant): 0.98 🎯
  - Recall (Malignant): 0.93 🎯
  - F1-Score (Malignant): 0.95 🎯
  - Precision (Benign): 0.96 🎯
  - Recall (Benign): 0.99 🎯
  - F1-Score (Benign): 0.97 🎯
- K-Nearest Neighbors (KNN) 👥:
  - Accuracy: 0.95 🎯
  - Precision (Malignant): 0.93 🎯
  - Recall (Malignant): 0.93 🎯
  - F1-Score (Malignant): 0.93 🎯
  - Precision (Benign): 0.96 🎯
  - Recall (Benign): 0.96 🎯
  - F1-Score (Benign): 0.96 🎯
- Optimized Models
- Optimized Random Forest 🌳:
  - Best Hyperparameters: criterion='entropy', max_depth=None, max_features='auto', n_estimators=200 🔧
  - Best Cross-Validation Score: 0.9714 🎯
  - Accuracy: 0.96 🎯
  - Precision (Malignant): 0.95 🎯
  - Recall (Malignant): 0.95 🎯
  - F1-Score (Malignant): 0.95 🎯
  - Precision (Benign): 0.97 🎯
  - Recall (Benign): 0.97 🎯
  - F1-Score (Benign): 0.97 🎯
- Optimized K-Nearest Neighbors (KNN) 👥:
  - Best Hyperparameters: metric='manhattan', n_neighbors=5, weights='uniform' 🔧
  - Best Cross-Validation Score: 0.9648 🎯
  - Accuracy: 0.96 🎯
  - Precision (Malignant): 0.95 🎯
  - Recall (Malignant): 0.95 🎯
  - F1-Score (Malignant): 0.95 🎯
  - Precision (Benign): 0.97 🎯
  - Recall (Benign): 0.97 🎯
  - F1-Score (Benign): 0.97 🎯

## 📈 Conclusions
- High Accuracy and Consistency: Both the initial and optimized models achieved high accuracy, with the Random Forest model consistently performing slightly better than the KNN model. The optimized models maintained or slightly improved the accuracy, demonstrating the robustness of the models.
- Precision and Recall: The precision and recall for both classes (Malignant and Benign) were high across all models. The optimized models showed balanced precision and recall, indicating that they are effective at correctly identifying both malignant and benign tumors without a significant trade-off between false positives and false negatives.
- F1-Score: The F1-scores for both classes were also high, reflecting a good balance between precision and recall. The optimized models slightly improved the F1-scores, particularly for the malignant class, which is crucial in medical diagnostics to ensure that malignant cases are not missed.
 - Hyperparameter Optimization: The use of GridSearchCV for hyperparameter optimization proved beneficial. For the Random Forest model, the best parameters included using the ‘entropy’ criterion and 200 estimators. For the KNN model, the best parameters were using the ‘manhattan’ distance metric with 5 neighbors and uniform weights. These optimizations helped in fine-tuning the models for better performance.
- Model Selection: While both models performed well, the Random Forest model had a slight edge in terms of overall performance metrics. However, the choice between these models can depend on other factors such as interpretability, computational efficiency, and specific use case requirements.

## 🎨 Visualizations
- Class Distribution
 - Description: A bar plot showing the distribution of malignant and benign cases in the dataset.
 - Purpose: To understand the balance of classes in the dataset, which is crucial for model training and evaluation.
- Correlation Matrix
 - Description: A heatmap displaying the correlation between different features in the dataset.
 - Purpose: To identify relationships between features, which can help in feature selection and understanding the data structure.
- Feature Distributions
 - Description: Histograms showing the distribution of key features like mean radius and mean texture.
 - Purpose: To visualize the spread and central tendency of important features, aiding in understanding their impact on the target variable.
- Boxplots by Class
 - Description: Boxplots comparing the distribution of features like mean radius and mean texture between malignant and benign classes.
 - Purpose: To compare the feature distributions across classes, highlighting differences that can be leveraged by the models.
- Pairplot of Selected Features
 - Description: A pairplot visualizing the relationships between selected features, colored by class.
 - Purpose: To explore the interactions between features and their combined effect on the target variable, providing insights into feature importance.

## 🗂️ Project Structure
- NoteBook

## 📬 Contact
- For any questions or suggestions, feel free to reach out via GitHub issues or contact me directly at jotaduranbon.com.


Happy coding! 😊
