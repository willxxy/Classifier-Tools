import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm

### MOCK DATA (Replace with real data)
n_samples = 1000

data = {
    'X': np.random.rand(n_samples),
    'Y': np.random.rand(n_samples),
    'Z': np.random.rand(n_samples),
    'Label': np.random.choice([0, 1, 2], n_samples)
}
df = pd.DataFrame(data)

df.fillna(df.mean(), inplace=True)

df['mean'] = df[['X', 'Y', 'Z']].mean(axis=1)
df['std'] = df[['X', 'Y', 'Z']].std(axis=1)
df['var'] = df[['X', 'Y', 'Z']].var(axis=1)
df['min'] = df[['X', 'Y', 'Z']].min(axis=1)
df['max'] = df[['X', 'Y', 'Z']].max(axis=1)

print(df.head())

X = df.drop('Label', axis=1).values
y = df['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

### KNN
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'KNN Accuracy: {accuracy}')
