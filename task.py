##CLASSIFICATION_SVM##
#importing the lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing the dataset
data_CLASS = pd.read_csv("Social_Network_Ads.csv")
#data preprocessing
X = data_CLASS.iloc[:, [2, 3]].values
y = data_CLASS.iloc[:, 4].values
data_CLASS.shape
data_CLASS.head()
data_CLASS.describe()
# splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#fitting the classification model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
#predicting the test set results
y_pred = classifier.predict(X_test)
#printting the confusion matrix for the model
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test,y_pred)
#visualizing the results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
################################################################################################
################################################################################################
################################################################################################
##REGRESSION_LINEAR_REGRESSION##
# importing the lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset
HouseDF = pd.read_csv('USA_Housing.csv')
HouseDF.head()  

# data preprocessing
HouseDF.info()
HouseDF.columns
X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y = HouseDF['Price']

# splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# fitting the regression model
from sklearn.linear_model import LinearRegression 
lm = LinearRegression() 
lm.fit(X_train,y_train) 
 
# predicting the test set results
predictions = lm.predict(X_test) 
 
# visualizing the results
plt.scatter(y_test,predictions)

#Regression Evaluation Metrics for the model
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions)) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#################################################################################################
#################################################################################################
#################################################################################################
##CLUSTERING_KMEAN##
#importing the lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing the dataset
data_CLUST=pd.read_csv("Mall_Customers.csv")
data_CLUST.head()
#data preprocessing
X = data_CLUST.iloc[:, [2, 4]].values
#fitting the clusttering model
from sklearn.cluster import KMeans
elbow=[]
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 101)
    kmeans.fit(X)
    elbow.append(kmeans.inertia_)
#showing the ELBOW METHOD results
import seaborn as sns
sns.lineplot(range(1, 20), elbow,color='blue')
plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
plt.title('ELBOW METHOD')
plt.show()
#predicting the test set results
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 101)
y_pred = kmeans.fit_predict(X)
#visualizing the results
plt.figure(figsize=(15,7.5))
sns.scatterplot(X[y_pred == 0, 0], X[y_pred == 0, 1],s=50)
sns.scatterplot(X[y_pred == 1, 0], X[y_pred == 1, 1],s=50)
sns.scatterplot(X[y_pred == 2, 0], X[y_pred == 2, 1],s=50)
sns.scatterplot(X[y_pred == 3, 0], X[y_pred == 3, 1],s=50)
sns.scatterplot(X[y_pred == 4, 0], X[y_pred == 4, 1],s=50)
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],s=500,color='yellow')
plt.title('Clusters')
plt.legend()
plt.show()


                     ######################################################
                                 ################################
                                        ####################