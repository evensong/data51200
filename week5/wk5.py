import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

data = pd.read_excel('HBAT(9).xls')
validation = pd.read_excel('HBAT _Test1.xls')

y_train = data.loc[:, 'x4']
x_train = data.iloc[:, 6:19]

y_test = validation.loc[:, 'x4']
x_test = validation.iloc[:, 6:19]

scale = StandardScaler()
scale.fit_transform(x_train)


lda = LinearDiscriminantAnalysis()
lda.fit(x_train.values, y_train.values)

train_pred = lda.predict(x_train)
test_pred = lda.predict(x_test)
train_confusion = confusion_matrix(y_train.values, train_pred)
test_confusion = confusion_matrix(y_test.values, test_pred)