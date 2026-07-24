import pandas as pd #for data
import matplotlib.pyplot as plt #for data visuals
from sklearn.model_selection import train_test_split  #for data devision
from sklearn.metrics import r2_score    #for accuracy check
from sklearn.linear_model import LinearRegression      #model selection 


data={
    "Study_Hours":[1,2,6,8,9,10,13,15,17,20,23],
    "Marks":[120,135,145,169,175,280,283,288,389,390,399]
}
df = pd.DataFrame(data)

#input
x=df[["Study_Hours"]]

#output
y=df["Marks"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()

model.fit(x_train,y_train)

predictions = model.predict(x_test)

score = r2_score(y_test,predictions)

print(round(score,2))

user = float(input("Enter the study hours: "))

prediction = model.predict([[user]])
print(f"predicted Marks,{prediction[0]:.2f }")