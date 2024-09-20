import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('titanic_data.csv')

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = data[features]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

feature_importances = model.feature_importances_
features = np.array(features)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importances')
plt.show()

#===================================================================================#

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

def get_user_input():
    Pclass = int(input("Enter Passenger Class (1, 2, or 3): "))
    Sex = int(input("Enter Sex (0 = Male, 1 = Female): "))
    Age = float(input("Enter Age: "))
    SibSp = int(input("Enter number of Siblings/Spouses Aboard: "))
    Parch = int(input("Enter number of Parents/Children Aboard: "))
    Fare = float(input("Enter Fare: "))
    Embarked = int(input("Enter Embarked (0 = S, 1 = C, 2 = Q): "))
    
    return np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])

def main():
    print("Titanic Survival Prediction")
    user_input = get_user_input()
    prediction = model.predict(user_input)
    output = 'Survived' if prediction[0] == 1 else 'Not Survived'
    print(f'The person would have {output}')

if __name__ == "__main__":
    main()
