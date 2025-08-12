import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree

data=pd.read_csv('result.csv')
df=pd.DataFrame(data)

x=df[['study_hours','attendance']]
y=df['result']
dtc=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtc.fit(x,y)
plt.figure(figsize=(8,6))
plot_tree(dtc,feature_names=['study_hours','attendance'],class_names=['Fail','Pass'],filled=True)
plt.show()

new=[[5,85]]
pred=dtc.predict(new)
print("Prediction for new student:","1" if pred[0]==1 else "0")
