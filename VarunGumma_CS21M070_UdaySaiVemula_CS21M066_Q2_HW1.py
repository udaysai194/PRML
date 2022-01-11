import numpy as np
import pandas as pd
from csv import reader, writer
import matplotlib.pyplot as plt
plt.style.use("seaborn")

############################################################ IMPORTANT METHODS ##########################################

def read_csv(key):
    fname = f"classification_{key}_data.csv"
    with open(fname) as csvfile:
        csv_reader = reader(csvfile, delimiter=',')
        csv_data = [row for row in csv_reader][1:]
        csv_data = [list(map(float, row)) for row in csv_data]
    csv_data = pd.DataFrame(csv_data)
    Y = np.array(csv_data[2])
    X = np.array(csv_data.drop(2, axis=1))
    return X, Y

def write_csv(X, Y, fname):
    with open(fname, 'w') as f:
        csv_writer = writer(f)
        csv_writer.writerow(["X1", "X2", "Y_pred"])
        csv_writer.writerows([[x1, x2, y] for ((x1, x2), y) in zip(X, Y)])

def confusion_matrix(Y, Y_pred):
    conf_mat = [[0, 0], [0, 0]]
    Y_pred[Y_pred < 0] = 0
    for (y, y_pred) in zip(Y, Y_pred):
        conf_mat[int(y)][int(y_pred)] += 1
    print(f"+{'-'*13}+")
    print(f"| {conf_mat[0][0] :^4} | {conf_mat[0][1] :^4} |")
    print(f"+{'-'*13}+")
    print(f"| {conf_mat[1][0] :^4} | {conf_mat[1][1] :^4} |")
    print(f"+{'-'*13}+") 

############################################################ CLASSES #####################################################

class Perceptron:
    def __init__(self):
        self.C = 0.
        self.gamma = 1.
        self.W = np.array([0., 1.])
    
    def fit_with_margins(self, X, Y, iters=1000):
        done = False
        for _ in range(iters):
            done = True
            for (x, y) in zip(X, Y):
                norm = np.linalg.norm(self.W)
                dist = (np.dot(x, self.W) + self.C)/norm
                if y == 0 and dist > -self.gamma:
                    self.W -= x
                    self.C -= 1
                    done = False
                elif y == 1 and dist < self.gamma:
                    self.W += x
                    self.C += 1
                    done = False
            if done:
                break
                     
    def predict(self, X):
        return np.sign(np.dot(X, self.W) + self.C)
        
    def plot(self, X, Y, key):
        norm = np.linalg.norm(self.W)
        C1 = self.C + self.gamma*norm
        C2 = self.C - self.gamma*norm
        plt.figure(figsize=(9,6))
        for ((x1, x2), y) in zip(X, Y):
            plt.scatter(x1, x2, color=('g' if y == 1 else 'r'), marker='*')
        x_ = np.linspace(-2, 13, 100)
        y_ = -(self.W[0]*x_ + self.C)/self.W[1]
        y1_ = -(self.W[0]*x_ + C1)/self.W[1]
        y2_ = -(self.W[0]*x_ + C2)/self.W[1]
        
        plt.plot(x_, y_, color='b')
        plt.plot(x_, y1_, color='b', ls="--")
        plt.plot(x_, y2_, color='b', ls="--")
        plt.title(f"Classified Data: Perceptron - {key}")
        plt.xlabel(xlabel=r"x$\rightarrow$", size=15)
        plt.ylabel(ylabel=r"y$\rightarrow$", size=15)
        plt.show()



class LinearDiscriminant:
    def __init__(self):
        self.intercept = None
        self.coef = None
        
    def fit(self, X, Y, cov="identity", alpha=2.0):
        feats = X.shape[1]
        X_pos = X[Y == 1]
        X_neg = X[Y != 1]
        n1 = X_pos.shape[0]
        n2 = X_neg.shape[0]
        m1 = np.mean(X_pos, axis=0)
        m2 = np.mean(X_neg, axis=0)
        if cov == "identity":
            s = (alpha ** 2) * np.identity(feats)
        elif cov == "first":
            s = np.cov(X_pos.T)
        elif cov == "second":
            s = np.cov(X_neg.T)
        s_inv = np.linalg.inv(s)
        self.intercept = -0.5*np.linalg.multi_dot([(m1 + m2).T, s_inv, (m1 - m2)]) + np.log(n1/n2)
        self.coef = np.dot(s_inv, (m1 - m2))
    
    def predict(self, X):
        return np.sign(self.intercept + np.dot(X, self.coef))
    
    def plot(self, X, Y, key):
        plt.figure(figsize=(9,6))
        for (x, c) in zip(X, Y):
            plt.scatter(x[0], x[1], color=('g' if c == 1. else 'r'), marker='*')
        x_ = np.linspace(-2, 13, 100)
        y_ = -(self.intercept + self.coef[0]*x_)/self.coef[1]
        plt.plot(x_, y_, color='b')
        plt.title(f"Classified Data: LD - {key}")
        plt.xlabel(xlabel=r"x$\rightarrow$", size=15)
        plt.ylabel(ylabel=r"y$\rightarrow$", size=15)
        plt.show()




class QuadraticDiscriminant:
    def __init__(self):
        self.A = None
        self.B = None
        self.C = None
        
    def fit(self, X, Y):
        feats = X.shape[1]
        X_pos = X[Y == 1]
        X_neg = X[Y != 1]
        n1 = X_pos.shape[0]
        n2 = X_neg.shape[0]
        m1 = np.mean(X_pos, axis=0)
        m2 = np.mean(X_neg, axis=0)
        s1 = np.cov(X_pos.T)
        s1_det = np.linalg.det(s1)
        s1_inv = np.linalg.inv(s1)
        s2 = np.cov(X_neg.T)
        s2_det = np.linalg.det(s2)
        s2_inv = np.linalg.inv(s2)
        
        self.A = 0.5*(s2_inv - s1_inv)
        self.B = np.dot(s1_inv, m1) - np.dot(s2_inv, m2)
        self.C = 0.5*np.linalg.multi_dot([m2.T, s2_inv, m2]) - \
                 0.5*np.linalg.multi_dot([m1.T, s1_inv, m1]) + \
                 0.5*np.log(s2_det/s1_det) + np.log(n1/n2)
        
    def predict(self, X):
        f = lambda x : (np.linalg.multi_dot([x.T, self.A, x]) + np.dot(x.T, self.B) + self.C)
        return np.array([np.sign(f(x)) for x in X])
    
    def plot(self, X, Y, key):
        plt.figure(figsize=(9,6))
        for (x, c) in zip(X, Y):
            plt.scatter(x[0], x[1], color=('g' if c == 1 else 'r'), marker='*')
            
        x_, y_ = np.meshgrid(np.arange(-5, 15, 0.025), 
                             np.arange(-5, 15, 0.025))
        
        plt.contour(x_, y_, 
                    x_*x_*self.A[0][0] + \
                    x_*y_*(self.A[0][1] + self.A[1][0]) + \
                    y_*y_*self.A[1][1] + \
                    x_*self.B[0] + \
                    y_*self.B[1] + \
                    self.C, 
                    [0], 
                    cmap="viridis")
        
        plt.title(f"Classified Data: QD - {key}")
        plt.xlabel(xlabel=r"x$\rightarrow$", size=15)
        plt.ylabel(ylabel=r"y$\rightarrow$", size=15)
        plt.show()


###################################################### DRIVER CODE ##################################################

X_train, Y_train = read_csv("train")
X_test, Y_test = read_csv("test")

print("Perceptron")
perceptron = Perceptron()
perceptron.fit_with_margins(X_train, Y_train)
print("confusion matrix for train data: ")
Y_pred_train = perceptron.predict(X_train)
confusion_matrix(Y_train, Y_pred_train)
print("\nconfusion matrix for test data: ")
Y_pred_test = perceptron.predict(X_test)
confusion_matrix(Y_test, Y_pred_test)
# perceptron.plot(X_train, Y_train, key="train")
perceptron.plot(X_test, Y_test, key="test")
write_csv(X_test, Y_pred_test, fname="perceptron_test.csv")


print("Linear Discriminant - 1")
lda = LinearDiscriminant()
lda.fit(X_train, Y_train, cov="identity", alpha=2.0)
print("confusion matrix for train data: ")
Y_pred_train = lda.predict(X_train)
confusion_matrix(Y_train, Y_pred_train)
print("\nconfusion matrix for test data: ")
Y_pred_test = lda.predict(X_test)
confusion_matrix(Y_test, Y_pred_test)
lda.plot(X_test, Y_test, key="test")
write_csv(X_test, Y_pred_test, fname="lda1_test.csv")


print("Linear Discriminant - 2")
lda = LinearDiscriminant()
lda.fit(X_train, Y_train, cov="first")
print("confusion matrix for train data: ")
Y_pred_train = lda.predict(X_train)
confusion_matrix(Y_train, Y_pred_train)
print("\nconfusion matrix for test data: ")
Y_pred_test = lda.predict(X_test)
confusion_matrix(Y_test, Y_pred_test)
lda.plot(X_test, Y_test, key="test")
write_csv(X_test, Y_pred_test, fname="lda2_test.csv")


qda = QuadraticDiscriminant()
qda.fit(X_train, Y_train)
print("confusion matrix for train data: ")
Y_pred_train = qda.predict(X_train)
confusion_matrix(Y_train, Y_pred_train)
print("\nconfusion matrix for test data: ")
Y_pred_test = qda.predict(X_test)
confusion_matrix(Y_test, Y_pred_test)
qda.plot(X_test, Y_test, key="test")
write_csv(X_test, Y_pred_test, fname="qda_test.csv")