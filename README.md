# **Alphabet Soup Charitable Donation Prediction - Deep Learning Model**

## **Overview**
The **Alphabet Soup Charitable Donation Prediction** project aims to predict the success of charitable donation applications based on various features. The dataset, provided by Alphabet Soup, contains information about different charity organizations and their respective application details. The model uses a deep learning approach with TensorFlow to classify whether a charity's application will be successful or not (binary classification: success = 1, failure = 0).

## **Objective**
The primary goal of this project is to:
- Build a deep learning model to predict the success of charitable donation applications.
- Preprocess and clean the dataset to make it suitable for training.
- Use TensorFlow and Keras to design, compile, and evaluate a deep neural network model.
- Achieve the best possible performance in predicting the success rate of charity applications.

## **Technologies Used**
- **Python**: Programming language used for data manipulation, model building, and evaluation.
- **TensorFlow/Keras**: Used to build and train the deep learning model.
- **Pandas**: Data manipulation and cleaning.
- **Scikit-learn**: Used for data preprocessing (splitting, scaling).
- **Google Colab**: Cloud-based platform used for running the code and training the model.
  
## **Installation**
To replicate this project, ensure you have the following Python libraries installed:
- TensorFlow
- Keras
- Scikit-learn
- Pandas

To install these packages, you can use the following pip commands:

```bash
pip install tensorflow
pip install pandas
pip install scikit-learn
```

## **Dataset**
The dataset used in this project is provided by Alphabet Soup and is available for download from the following URL:

[Alphabet Soup Charitable Data](https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv)

The dataset includes the following columns:
- **EIN**: Employer Identification Number (non-beneficial)
- **NAME**: Name of the charity (non-beneficial)
- **APPLICATION_TYPE**: Type of application (categorical)
- **AFFILIATION**: Type of affiliation (categorical)
- **CLASSIFICATION**: Charity classification (categorical)
- **USE_CASE**: Purpose of the charity (categorical)
- **ORGANIZATION**: Organization type (categorical)
- **STATUS**: Status of the application (binary)
- **INCOME_AMT**: Income amount (categorical)
- **SPECIAL_CONSIDERATIONS**: Special considerations (binary)
- **ASK_AMT**: Amount requested (numeric)
- **IS_SUCCESSFUL**: Target variable (binary - 1 for success, 0 for failure)

## **How to Use**
To run the model, follow these steps:
1. Clone the repository.
2. Install the required dependencies using the provided pip commands.
3. Load the dataset and perform preprocessing as described.
4. Define the model, compile it, and train it on the preprocessed data.
5. Evaluate the model using the test set.
6. Optionally, modify hyperparameters and test other machine learning models.

## REPORT
### Overview of the Analysis:

The purpose of this analysis was to build a deep learning model using TensorFlow to predict the success of charitable donations based on various features. This analysis leverages a dataset from Alphabet Soup, which includes information about charity organizations and their donation-related features. The goal is to predict whether a charity’s application will be successful (binary classification problem).

### Results:

#### Data Preprocessing

- **Target Variables:**
  - The target variable for the model is **`IS_SUCCESSFUL`**, which indicates whether a charity’s application for funding is successful or not (1 for success, 0 for failure).

- **Feature Variables:**
  - The features used in the model are derived from the remaining columns after removing irrelevant information. Although some of these variables were removed in optimization tests, the final list of feature variables in AlphabetSoupCharity_Optimization.ipynb is:
    - `APPLICATION_TYPE`
    - `AFFILIATION`
    - `CLASSIFICATION`
    - `USE_CASE`
    - `ORGANIZATION`
    - `STATUS`
    - `INCOME_AMT`
    - `SPECIAL_CONSIDERATIONS`
    - `ASK_AMT`
  
  These variables represent different categorical and numerical attributes of each charity application.

- **Variables to Remove:**
  - **`EIN`** and **`NAME`** were removed as these variables were non-beneficial for the analysis. The EIN is a unique identifier for each charity, and the NAME of the charity does not add any predictive value to the model.

#### Compiling, Training, and Evaluating the Model

- **Model Architecture:**
  - The neural network consists of 4 layers:
    1. **Input Layer**: This layer has 46 neurons (equal to the number of features in the dataset).
    2. **First Hidden Layer**: 64 neurons with ReLU activation.
    3. **Second Hidden Layer**: 32 neurons with ReLU activation.
    4. **Third Hidden Layer**: 16 neurons with ReLU activation.
    5. **Output Layer**: 1 neuron with a sigmoid activation function to output the probability of success (binary classification).

  **Reasoning for architecture selection:**
  - The three hidden layers with decreasing neuron counts are chosen to allow the model to learn increasingly abstract features while reducing the complexity of the network.
  - **ReLU** activation functions were selected for the hidden layers due to their ability to help the network converge faster and mitigate vanishing gradient issues. During earlier optimization tests, **swish** activation was tested, but did not improve the model's accuracy.
  - The **sigmoid** activation in the output layer is appropriate for binary classification, as it maps outputs to a range between 0 and 1, representing the probability of success.

- **Model Performance:**
  - The model was not able to achieve the target model performance of 75%.
  - **Achieved Performance:**
    - The final model achieved an accuracy of **72.96%** on the test data, with earlier optimization tests resulting in rates just above 73%.
    - The loss function used was **binary crossentropy**, and the optimizer used was **Adam**.

- **Steps Taken to Increase Model Performance:**
  - **Data Preprocessing:**
    - Categorical variables were converted to numeric using one-hot encoding (`pd.get_dummies`).
    - Low-frequency categories in `APPLICATION_TYPE` and `CLASSIFICATION` were replaced with "Other" to reduce model complexity and noise.
    - Standardization of the features was performed using `StandardScaler` to ensure that the model performed well on the scaled data, reducing bias caused by features with large numerical ranges.
  
  - **Model Optimization:**
    - The model architecture was experimented with by adding a third hidden layer to enhance its ability to learn complex patterns.
    - A relatively higher number of epochs (25) were used to train the model, ensuring enough training time to learn the patterns in the data. However, adjustments to the epochs or batch sizes could further improve performance.
    - optimization tests were performed using **swish** activation, using a higher number of epochs, and removing more of the initial df columns, all with limited or negative effects on the model's accuracy.

#### Summary:

- **Overall Results:**
  - The deep learning model achieved an accuracy of 72.96%, which is a reasonable performance given the complexity of the data. The model did not reach the ideal performance target of 75%, but it performed adequately given the preprocessing efforts and the model’s structure.

- **Recommendations for Improving the Model:**
  - **Hyperparameter Tuning**: Further optimization could be done by adjusting the learning rate, batch size, or the number of neurons per layer. Using techniques such as grid search or random search for hyperparameter tuning could help find the best combination for improved performance.
  - **Alternative Models**: For a classification problem like this, other machine learning models such as Random Forest or Gradient Boosting (e.g., XGBoost) might yield better results. These models often perform well in tabular data with categorical variables. Moreover, using ensemble methods could help increase the prediction accuracy.
  - **Cross-Validation**: Implementing k-fold cross-validation during training can help identify the model's true performance by reducing overfitting and ensuring that the model generalizes better on unseen data.