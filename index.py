import pandas as pd
import numpy as np
from sklearn.utils import resample 
from sklearn.cluster import KMeans

df = pd.read_csv('Creditcard_data.csv')

# Separate majority and minority classes
df_majority = df[df.Class==0]
df_minority = df[df.Class==1]
# Upsample minority class
df_minority_upsampled = resample(df_minority, replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123) # reproducible results
# Combine majority class with upsampled minority class
df_balanced = pd.concat([df_majority, df_minority_upsampled])
# Display new class counts
print(df_balanced.Class.value_counts())

#Sampling
Sampling1 = df_balanced.sample(frac=0.5, replace=True, random_state=1) 
Sampling2 = df.groupby('Class', group_keys=False).apply(lambda x: x.sample(frac=0.6))
def systematic_sampling(df, step):
    indexes = np.arange(0, len(df_balanced), step=step)
    systematic_sample = df.iloc[indexes]
    return systematic_sample
Sampling3 = systematic_sampling(df_balanced , 3)
#print(Sampling3)

def get_clustered_Sample(df, n_per_cluster, num_select_clusters):
    N = len(df)
    K = int(N/n_per_cluster)
    data = None
    for k in range(K):
        sample_k = df.sample(n_per_cluster) 
        sample_k["cluster"] = np.repeat(k,len(sample_k))  
        df = df.drop(index = sample_k.index) 
        data = pd.concat([data,sample_k],axis = 0) 
    random_chosen_clusters = np.random.randint(0,K,size = num_select_clusters) 
    samples = data[data.cluster.isin(random_chosen_clusters)]
    return(samples)
Sampling4 = get_clustered_Sample(df = df_balanced, n_per_cluster = 850, num_select_clusters = 2)
Sampling5 = df_balanced.groupby('Class', group_keys=False).apply(lambda x: x.sample(300))
#print(Sampling5)

#Model training & testing 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

X = Sampling1.drop('Class', axis=1)
y = Sampling1['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
acc1=[]
for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    acc1.append(round(training_score.mean(), 2) * 100)
    #print("Classifiers: ", classifier._class.__name_, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

X = Sampling2.drop('Class', axis=1)
y = Sampling2['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
acc2=[]
for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    acc2.append(round(training_score.mean(), 2) * 100)
    #print("Classifiers: ", classifier._class.__name_, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

X = Sampling3.drop('Class', axis=1)
y = Sampling3['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
acc3=[]
for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    acc3.append(round(training_score.mean(), 2) * 100)
    #print("Classifiers: ", classifier._class.__name_, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

X = Sampling4.drop('Class', axis=1)
y = Sampling4['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
acc4=[]
for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    acc4.append(round(training_score.mean(), 2) * 100)
    #print("Classifiers: ", classifier._class.__name_, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

acc5=[]
X = Sampling5.drop('Class', axis=1)
y = Sampling5['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    acc5.append(round(training_score.mean(), 2) * 100)
    #print("Classifiers: ", classifier._class.__name_, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
    
row_names = ['M1' , 'M2' , 'M3' , 'M4'];
col_names=  ['Models' , 'Sampling1' , 'Sampling2' , 'Sampling3' , 'Sampling4' , 'Sampling5']
d = {
     'Model' : ['M1' , 'M2' , 'M3' , 'M4'],
     'Sampling1': acc1,
     'Sampling2': acc2,
     'Sampling3': acc3,
     'Sampling4': acc4,
     'Sampling5': acc5
 }

ans_df = pd.DataFrame(data=d)
print(ans_df)