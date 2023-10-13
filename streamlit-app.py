import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
# Define adjustable parameters for each model
model_params = {
    'Logistic Regression (baseline)': {'max_iter': 100000, 'random_state': 0},
    'Gradient Boosting Classifier': {'random_state': 0, 'n_estimators': 100, 'learning_rate': 0.1},
    'Ada Boosting Classifier': {'random_state': 0, 'n_estimators': 50, 'learning_rate': 1.0},
    'Random Forest Classifier': {'random_state': 0, 'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2},
    'Support Vector Machine': {'C': 1.0, 'kernel': 'rbf'},
    'Decision Tree Classifier': {'random_state': 0, 'max_depth': 10, 'min_samples_split': 2},
    'Bagging Classifier': {'random_state': 0, 'n_estimators': 100},
    'KNeighbors Classifier': {'n_neighbors': 5}
}

models = {}

# Create model instances with adjustable parameters
for model_name, params in model_params.items():
    if model_name == 'Logistic Regression (baseline)':
        models[model_name] = LogisticRegression(**params)
    elif model_name == 'Gradient Boosting Classifier':
        models[model_name] = GradientBoostingClassifier(**params)
    elif model_name == 'Ada Boosting Classifier':
        models[model_name] = AdaBoostClassifier(**params)
    elif model_name == 'Random Forest Classifier':
        models[model_name] = RandomForestClassifier(**params)
    elif model_name == 'Support Vector Machine':
        models[model_name] = SVC(**params)
    elif model_name == 'Decision Tree Classifier':
        models[model_name] = DecisionTreeClassifier(**params)
    elif model_name == 'Bagging Classifier':
        models[model_name] = BaggingClassifier(**params)
    elif model_name == 'KNeighbors Classifier':
        models[model_name] = KNeighborsClassifier(**params)
    elif model_name == 'MLP Classifier':
        models[model_name] = MLPClassifier(**params)

# Load your dataset for EDA
@st.cache
def load_data():
    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)
    # data (as pandas dataframes)
    x_unsanitized = heart_disease.data.features
    y_unsanitized = heart_disease.data.targets
    X_df = pd.DataFrame(x_unsanitized, columns=heart_disease.feature_names)
    y_series = pd.Series(y_unsanitized["num"], name="target")

    # Merge X and y
    merged_df = pd.concat([X_df, y_series], axis=1)
    return merged_df.dropna()

# EDA Section
st.title('Exploratory Data Analysis (EDA)')
data = load_data()

# Display the dataset
st.subheader('Raw Data Cleaned')
st.write(data)

# Add controls for EDA (e.g., plot options, filters, etc.)
# You can use various st components for interactivity

# Example: Histogram for numeric columns
st.subheader('Histogram for Numeric Columns')
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
selected_col = st.selectbox('Select a numeric column for histogram:', numeric_cols)

# Create a histogram using Matplotlib
fig, ax = plt.subplots()
ax.hist(data[selected_col], bins=20, edgecolor='k')
ax.set_xlabel(selected_col)
ax.set_ylabel('Frequency')
st.pyplot(fig)  # Display the Matplotlib figure with st.pyplot

# Example: Correlation Heatmap
st.subheader('Correlation Heatmap')
st.write("Correlation heatmap of numeric columns:")
st.write(data[numeric_cols].corr())

# Machine Learning Section
st.title('Machine Learning Algorithms')

# Sidebar controls for model selection and hyperparameter tuning
selected_model = st.sidebar.selectbox('Select a Machine Learning Model', list(models.keys()))

# Allow users to modify hyperparameters for the selected model
st.sidebar.header(f'{selected_model} Hyperparameters')
params = {}
for param_name, param_value in model_params[selected_model].items():
    if isinstance(param_value, int):
        params[param_name] = st.sidebar.slider(param_name, 1, 100, param_value)
    elif isinstance(param_value, float):
        params[param_name] = st.sidebar.slider(param_name, 0.01, 10.0, param_value)
    else:
        params[param_name] = st.sidebar.text_input(param_name, param_value)

# Split data into features and labels
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train and evaluate the selected model
st.subheader(f'{selected_model} Classifier')
model = models[selected_model]
model.set_params(**params)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
st.write('Accuracy:', "Accuracy: {:.2f}%".format(accuracy * 100))

