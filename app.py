import uvicorn
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

housing = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv")

housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)
housing_num = housing.drop("ocean_proximity", axis=1)

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer    # data cleaning
from sklearn.preprocessing import OneHotEncoder # para tratar dados textuais
from sklearn.preprocessing import StandardScaler # feature scaling

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

# Completo, para dados não numéricos e numéricos
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

model = joblib.load("Housing_RandomForest.pkl")

def get_price(data):
  data = np.array([data])
  data = pd.DataFrame(data, columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms','total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'])
  prepared = full_pipeline.transform(data)
  price = model.predict(prepared)

  return price

# inciar app
app = FastAPI(title="California Housing Prices API", description="API for California housing prices predicting using ML", version="1.0")

# criar rota
@app.get('/')
async def index():
  return {"text":"Hello"}

@app.get('/predict/{longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value, ocean_proximity}')

async def predict(longitude, latitude, housing_median_age, total_rooms, 
                    total_bedrooms, population, households, median_income, 
                    ocean_proximity):

  price = get_price([float(longitude), float(latitude), float(housing_median_age), float(total_rooms), float(total_bedrooms), float(population), float(households), float(median_income), str(ocean_proximity)])
  
  return {"price":price[0]}

if __name__ == '__main__':
  uvicorn.run(app, host="127.0.0.1", port=8000)