import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df=pd.read_csv('ml3.csv')
df['Gender']=LabelEncoder().fit_transform(df['Gender'])

x=df[['Age','Gender','BMI','BP','Cholesterol']]
y=df['condition']
scaler=StandardScaler()
xscale=scaler.fit_transform(x)
xtr,xte,ytr,yte=train_test_split(xscale,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(xtr,ytr)
ypr=model.predict(xte)
yprob=model.predict_proba(xte)[:,1]
print('Accuracy',accuracy_score(yte,ypr))
print('Classification Report:\n',classification_report(yte,ypr,zero_division=1))
#print("Confusion Matrix:\n",confusion_matrix(yte,ypr))
cm=confusion_matrix(yte,ypr)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

new=pd.DataFrame([[60,1,27,130,200]],columns=['Age','Gender','BMI','BP','Cholesterol'])
newscale=scaler.transform(new)
newcondition=model.predict_proba(newscale)[0][1]
print(f"Probability of developing the condition:{newcondition:.2f}")
