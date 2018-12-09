# Importing the libraries
import numpy as np # for numerical, stats, matrix operation
import matplotlib.pyplot as plt # for plotting
import pandas as pd # for data manipulation

def df_info(df): # function to report some general info of a dataframe 
	print("Printing general info:\n")
	print(df.columns) 
#	print(df.dtypes) 
	print(df.shape)
	print("Null check:\n")
	nullcheck = pd.isnull(df).sum() 
	print(nullcheck[nullcheck>0])

# Importing the dataset
#col_names = ["ID","Diagnosis","radius","texture","perimeter","area","smoothness","compactness","concavity","cancave points","symmetry","fractal"]
df = pd.read_csv(r'diabetes.csv')

df_info(df)

#df['Glucose'][df['Glucose'] == 0] = np.nan
#df['Glucose'] = df['Glucose'].apply(lambda x: np.nan if x == 0.0) # tag
#df = df.dropna(axis = 0) # dropping the rows

# applying a lambda function onto a pandas Series

#def test(x): # x will just be one element in the Series
#    return 1

#df['Glucose'] = df['Glucose'].apply(lambda x:1)
#df.iloc[0,:] = df.iloc[0,:].apply(lambda x:1)

# Now, I want to apply a function to a dataframe 

def subbymean(col_in): # function takes in a whole column
    col = col_in.copy()
    col_nonz = col[col != 0]
    zero_c = col[col == 0].count()
    real_mean = col_nonz.mean()
    real_std = col_nonz.std()
    random_list = np.random.uniform(real_mean - real_std, real_mean + real_std, size=zero_c)
    col[col == 0] = random_list
#    col[col == 0] = real_mean
    return col

def test(row):
    return row

cols_to_substitute = ['Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
df.loc[:,cols_to_substitute] = df.loc[:,cols_to_substitute].apply(subbymean, axis=0) # apply this function to a column

X = df.loc[:,['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values # Features
#X = df.iloc[:,:8].values # Features
y = df.iloc[:,8].values # outcome that we want to predict
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# 25% will be test set, 75% will be training set

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC # SVC support vector classifier

clf = SVC(kernel = 'rbf',random_state = 0).fit(X_train, y_train) # hyperparameters to be tuned
y_pred = clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#print(np.mean((y_test == clf.predict(X_test))))
#print(np.mean((y_train == clf.predict(X_train))))

y_pred_fake = np.zeros(192)

from sklearn.model_selection import cross_val_score # this is for cross-validation
scores = cross_val_score(clf, X_train, y_train, cv=10) # 10 partitions
print(scores.mean(),scores.std())

print("fake model accuracy: {}".format((y_test == y_pred_fake).mean()))

from sklearn.metrics import roc_auc_score, accuracy_score
auc_score = roc_auc_score(y_test, y_pred, average='macro')
auc_score_fake = roc_auc_score(y_test, y_pred_fake, average='macro')
print("real model AUC: {}".format(auc_score))
print("fake model AUC: {}".format(auc_score_fake))

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

knn = KNeighborsClassifier(n_neighbors = 5)
nb = GaussianNB()
svc = SVC()
lr = LogisticRegression()
DT = DecisionTreeClassifier()
RF = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
'''
# GS for DT
param_grid = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_depth': [2,3,4,5,6,7,8],
              'min_samples_split': [2,3,4,5,6,7,8],
              'min_samples_leaf': [1,2,3,4,5,6]}
grid_search = GridSearchCV(estimator = DT,
                           param_grid = param_grid,
                           scoring = 'accuracy',
                           cv = 10, 
                           n_jobs = -1) 
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
print(best_parameters)
DT = DecisionTreeClassifier(**best_parameters)
'''

'''
# GS for RF
param_grid = {'max_depth': [2,3,4,5,6,7,8],
              'min_samples_split': [2,3,4,5,6,7,8],
              'min_samples_leaf': [1,2,3,4,5,6]}
grid_search = GridSearchCV(estimator = RF,
                           param_grid = param_grid,
                           scoring = 'accuracy',
                           cv = 10, 
                           n_jobs = -1) 
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
print(best_parameters)
RF = DecisionTreeClassifier(**best_parameters)
'''

'''
# Note: here is how you perform the gsCV, SVC
parameters = [{'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = svc,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10, 
                           n_jobs = -1) 
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
svc = SVC(**best_parameters)
'''


models = {"knn":knn, "nb":nb, "svc":svc, "LR": lr, "DT": DT, "RF": RF}

for name, model in models.items():
    print("\n"+name+":\n")
    model.fit(X_train,y_train) # loads the data into the model and fit it
    y_pred = model.predict(X_test)
    print("Accuracy score: {}".format(accuracy_score(y_test, y_pred)))
    print("AUC score: {}".format(roc_auc_score(y_test, y_pred, average='macro')))
