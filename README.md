
**Project: Student Academic Trajectory Prediction using Neural Networks**

**Purpose:**

This project aims to predict a student's academic trajectory through a university program by analyzing their course enrollment history. The core of the project is a neural network trained on previous student data to predict the likelihood of students taking specific courses in future semesters. 

**Dataset:**

* **DataSet.csv:** Contains historical student enrollment data in a tabular format. Columns represent:
    * **ID:** Unique student identifier.
    * **Semestre:** The semester number.
    * **Course Names:** Individual course names as columns with binary values (1 = enrolled, 0 = not enrolled). 

**Code Explanation**

**1. Libraries and Data Import**

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn as sk
import sklearn.neural_network

# ... Other imports
```

* **numpy:**  For numerical operations and array handling.
* **matplotlib.pyplot:** Data visualization (optional).
* **pandas:** For efficient data manipulation and loading the CSV.
* **sklearn:** Provides machine learning tools, including:
    * **neural_network:** Implementing neural network models.
    * **Other imports (as needed):** Preprocessing tools or additional ML algorithms.

**2. Loading the Dataset**

```python
df = pd.read_csv(r"/content/DataSet.csv")
df.head()
```

* Loads the DataSet.csv using `pd.read_csv()`.
* `head()` displays the first few rows to preview the data.

**2. Data Preprocessing**

```python
data = np.array(df.values)

s = 50 #semestres
m = 49 #materias
In_train = np.zeros((s, m)) #(semestres, materias)
Out_train = np.zeros((s, m)) #(semestres, materias)

# ... Preprocessing steps
```

* **Data Transformation:** Converts the DataFrame into a NumPy array.
* **Defining Variables:** Sets the number of semesters and courses.
* **Creating Input/Output Arrays:** Initializes arrays to hold input and output data for the model.
* **Preprocessing Logic:**
    * Separates enrolled courses into input arrays.
    * Calculates the difference between semesters as the target output (which courses are newly taken).

**3. Model Training**

```python
X_train = pd.DataFrame(In_train)
Y_train = pd.DataFrame(Out_train)

neural_net.fit(X_train, Y_train)
```

* **Data Formatting:** Converts input/output arrays back to DataFrames for compatibility with the model.
* **Model Creation:**  Instantiates a neural network model (e.g., `sklearn.neural_network.MLPClassifier` or `MLPCRegressor`).
* **Training:** Calls the `fit()` method to train the neural network model on the prepared dataset.

**4. Prediction**

```python
df_pred = pd.DataFrame({
    "Calculo 1":              [1],
    "Algebra Lineal":         [0],
    # ... Other courses
})

prediction = neural_net.predict(df_pred)  
print(prediction)
```

* **New Student Data:** Creates a DataFrame representing the current courses a student is taking.
* **Prediction:** Uses the trained model's `predict()` method to predict likely courses for future semesters.
* **Output:** The prediction will be an array indicating probabilities or classifications for future course enrollment.
