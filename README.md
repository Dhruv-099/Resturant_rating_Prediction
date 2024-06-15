# Resturant_rating_Prediction
 
Sure, here's a README file for the restaurant rating prediction app:

# Restaurant Rating Prediction App

This machine learning project predicts a restaurant's rating based on various features such as average cost for two, table booking availability, online delivery option, and price range. The dataset used for training the model is obtained from Kaggle, and the app is deployed using Streamlit, a Python library for building interactive web applications.

## Dataset

The dataset used for this project is the "Restaurant Ratings" dataset from Kaggle. It contains various restaurants' cuisine types, average cost for two, ratings, votes, and other details. The dataset can be found [here](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants).

## Requirements

To run this project, you need to have the following dependencies installed:

- Python 3.11.9
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit


## Project Structure

- `Resturant_Rating.ipynb`: This Jupyter Notebook contains the code for data exploration, preprocessing, feature engineering, model training, and evaluation.
- `app.py`: This Python script contains the code for the Streamlit app, which takes user input and predicts the restaurant rating class based on the trained model.
- `mlmodel.pkl`: This file contains the trained machine learning model, which is loaded by the Streamlit app for making predictions.
- `Scaler.pkl`: This file contains the scaler object used for scaling the input features before making predictions.

## Model Training

The `Resturant_Rating.ipynb` notebook covers data exploration, preprocessing, feature engineering, model training, and evaluation. Various machine learning algorithms, such as Linear Regression, Support Vector Machines (SVMs), Decision Trees, Random Forests, K-nearest neighbours (KNN), and AdaBoost, were explored and evaluated using cross-validation techniques.

The best-performing model (Random Forest Regressor) was selected and saved as `mlmodel.pkl` for deployment in the Streamlit app.

## Deployment

The Streamlit app (`app.py`) loads the trained model (`mlmodel.pkl`) and the scaler object (`Scaler.pkl`) to make predictions based on user input. The user can enter the required features, and the app will display the predicted rating class for the restaurant.
