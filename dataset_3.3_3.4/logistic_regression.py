import numpy as np  			
import pandas as pd				
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

np.random.seed(1234)

path = "data2.txt"

data = pd.read_csv(path, names=['Exam 1', 'Exam 2', 'Accepted'])

gamma = 0.9
eps = 1e-10
beta_1 = 0.9
beta_2 = 0.999

def get_Xy_theta(data):
    data.insert(0,'$x_0$',1)
    cols = data.shape[1]

    X_ = data.iloc[:,0:cols-1]
    X = X_.values

    y_ = data.iloc[:,cols-1:cols]
    y = y_.values   

    return X,y

X,y = get_Xy_theta(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=1234)
# print(X)
# print(X.shape)
# print(y.shape)
# print(theta.shape)

def sigmoid(z):    
	return 1 / (1 + np.exp(-z))

def loss_function(X,y,theta):
	A = sigmoid(np.matmul(X, theta))
	first = y * np.log(A)
	second = (1-y) * np.log(1-A)
	return -np.sum(first + second) / len(X) 


def Minimize(X, y, X_valid, y_valid, method="SGD"):
    m = X.shape[0]
    theta = np.zeros((X.shape[1], y.shape[1]))
    best_valid_acc = 0.0
    best_valid_theta = np.zeros_like(theta)
    best_epoch = 0
    best_time = 0
    momentum = np.zeros_like(theta)
    v = np.zeros_like(theta)       
    valid_x_range = []
    valid_acc_range = []      
    loss_range = []
    x_range = []            
    start_time = time.time()      
    
    for epoch in tqdm(range(epoch_num)):
        A = sigmoid(np.matmul(X, theta))
        grad_theta = np.matmul(X.T, A - y) / m
        
        loss = loss_function(X,y,theta)

        if (method == "SGD"):
            theta = theta - lr * grad_theta

        if (method == "SGD_with_momentum"):
            momentum = gamma * momentum + lr * grad_theta
            theta -= momentum
        
        if (method == "SGD_with_nesterov_momentum"):
            B = sigmoid(np.matmul(X, theta - gamma * momentum))
            momentum = gamma * momentum + lr / m * np.matmul(X.T, B - y)
            theta -= momentum

        if (method == "Adagrad"):
            lr_adagrad = 0.1
            v += grad_theta ** 2
            theta -= lr_adagrad / np.sqrt(v + eps) * grad_theta

        if (method == "RMSprop"):
            lr_rmsprop = 0.01
            v = gamma * v + (1.0 - gamma) * (grad_theta ** 2)
            theta -= lr_rmsprop / np.sqrt(v + eps) * grad_theta

        if (method == "Adam"):
            lr_adam = 0.01
            momentum = beta_1 * momentum + (1 - beta_1) * grad_theta
            v = beta_2 * v + (1 - beta_2) * (grad_theta ** 2)
            momentum_hat = momentum / (1.0 - np.power(beta_1, i+1))
            v_hair = v / (1.0 - np.power(beta_2, i+1))
            theta -= lr_adam * momentum_hat / (np.sqrt(v_hair) + eps)
        

        if (method == "AdaMax"):    
            momentum = beta_1 * momentum + (1 - beta_1) * grad_theta
            momentum_hat = momentum / (1.0 - np.power(beta_1, i+1))
            v = np.maximum(beta_2 * v, np.abs(grad_theta))
            theta -= lr * momentum_hat / (v + eps)

        if (method == "NAdam"):
            momentum = beta_1 * momentum + (1 - beta_1) * grad_theta
            momentum_hat = momentum / (1.0 - np.power(beta_1, i+1))
            momentum_bar = (beta_1 * momentum_hat + (1 - beta_1) * grad_theta) / (1.0 - np.power(beta_1, i+1))
            v = beta_2 * v + (1 - beta_2) * (grad_theta ** 2)
            v_hair = v / (1.0 - np.power(beta_2, i+1))
            theta -= lr * momentum_bar / (np.sqrt(v_hair) + eps)


        if (method == "Newton"):     
            alpha = 0.01
            beta = 0.5
            A_ = np.repeat(A, X.shape[1], 1)
            A2_ = 1.0 - A_
            XA = X * A_
            XA2 = X * A2_
            H = np.matmul(XA.T, XA2) / m
            d = np.matmul(np.linalg.inv(H), -grad_theta)
            lm = np.matmul(grad_theta.T, -d)
            t = 1
            while (loss_function(X, y, theta) - loss_function(X, y, theta+t*d) < alpha * t * lm):
                t = beta * t
            theta += t * d
            if (lm * lm <= 2 * eps):
                break

            # theta += lr * d
        
        if (method == "Newton_CG"):
            alpha = 0.01
            beta = 0.5
            max_iter = 1000
            A_ = np.repeat(A, X.shape[1], 1)
            A2_ = 1.0 - A_
            XA = X * A_
            XA2 = X * A2_
            H = np.matmul(XA.T, XA2) / m
            d = 0
            ng = np.linalg.norm(grad_theta)
            CG_tol = min(0.5, ng) * ng
            r = grad_theta
            p = -r
            for j in range(max_iter):
                rr = np.linalg.norm(r) ** 2
                Ap = np.matmul(H, p)
                alpha_ = rr / np.matmul(p.T, Ap)
                d = d + alpha_ * p
                r = r + alpha_ * Ap
                nr1 = np.linalg.norm(r)
                if (nr1 <= CG_tol):
                    break
                beta_ = nr1 ** 2 / rr
                p = -r + beta_ * p
                
            lm = np.matmul(grad_theta.T, -d)
            t = 1
            while (loss_function(X, y, theta) - loss_function(X, y, theta+t*d) < alpha * t * lm):
                t = beta * t
            theta += t * d
            if (lm * lm <= 2 * eps):
                break

            # theta += lr * d
        
        if (method == "Hessian_free"):
            alpha = 0.01
            beta = 0.5
            max_iter = 1000
            d = 0
            ng = np.linalg.norm(grad_theta)
            CG_tol = min(0.5, ng) * ng
            r = grad_theta
            p = -r            
            A1 = sigmoid(np.matmul(X, theta))            
            grad1 = np.matmul(X.T, A1 - y) / m
            for j in range(max_iter):
                rr = np.linalg.norm(r) ** 2
                A2 = sigmoid(np.matmul(X, theta+eps*p))
                grad2 = np.matmul(X.T, A2 - y) / m
                Ap = (grad2 - grad1) / eps
                alpha_ = rr / np.matmul(p.T, Ap)
                d = d + alpha_ * p
                r = r + alpha_ * Ap
                nr1 = np.linalg.norm(r)
                if (nr1 <= CG_tol):
                    break
                beta_ = nr1 ** 2 / rr
                p = -r + beta_ * p
                
            lm = np.matmul(grad_theta.T, -d)
            t = 1
            while (loss_function(X, y, theta) - loss_function(X, y, theta+t*d) < alpha * t * lm):
                t = beta * t
            theta += t * d
            if (lm * lm <= 2 * eps):
                break

            # theta += lr * d

        if (method == "BFGS" or method == "BGFS_momentum"):  
            max_iter = 1000
            alpha = 0.01
            beta = 0.5
            B = np.eye(theta.shape[0])
            if (np.linalg.norm(grad_theta) <= eps):
                break
            d = np.matmul(np.linalg.inv(B), -grad_theta)
            lm = np.matmul(grad_theta.T, -d)
            t = 1
            while (loss_function(X, y, theta) - loss_function(X, y, theta+t*d) < alpha * t * lm):
                t = beta * t

            if (method == "BFGS"):
                theta += t * d
                s = t * d

            if (method == "BFGS_momentum"):
                momentum = gamma * momentum + t * d
                theta += momentum
                s = momentum


            # if (method == "BFGS"):
            #     theta += lr * d
            #     s = lr * d 
            
            # if (method == "BFGS_momentum"):
            #     momentum = gamma * momentum + lr * d
            #     theta += momentum
            #     s = momentum
            current_grad_theta = np.matmul(X.T, sigmoid(np.matmul(X, theta)) - y) / m
            yy = current_grad_theta - grad_theta
            B = B - np.matmul(np.matmul(B, s), np.matmul(s.T, B)) / np.matmul(np.matmul(s.T, B), s) + np.matmul(yy, yy.T) / np.matmul(yy.T, s)

            if (lm * lm <= 2 * eps):
                break

        #print(f"epoch {epoch}, train loss:{loss:.4f}")
        if (method == "Newton" or method == "Newton_CG" or method == "Hessian_free"):
            valid_acc = predict(theta, X_valid, y_valid)
            valid_acc_range.append(valid_acc)
            valid_x_range.append(epoch)
            if (valid_acc > best_valid_acc):
                best_valid_acc = valid_acc
                best_valid_theta = theta
                best_time = time.time() - start_time
                best_epoch = epoch
        else:
            if (epoch % 100 == 0):
                valid_acc = predict(theta, X_valid, y_valid)
                valid_acc_range.append(valid_acc)
                valid_x_range.append(epoch)
                if (valid_acc > best_valid_acc):
                    best_valid_acc = valid_acc
                    best_valid_theta = theta
                    best_time = time.time() - start_time
                    best_epoch = epoch

    return best_valid_theta, x_range, loss_range, valid_x_range, valid_acc_range, best_time, best_epoch

lr = 0.001
epoch_num = 100000


def predict(theta, X, y):
    prob = sigmoid(np.matmul(X, theta))
    h_x = [1 if result >= 0.5 else 0 for result in prob]
    h_x = np.array(h_x).reshape(len(h_x), 1)
    acc = np.mean(h_x == y)
    return acc


first_order_method = ["SGD", "SGD_with_momentum", "SGD_with_nesterov_momentum", "Adagrad", "RMSprop", "Adam", "AdaMax", "NAdam"]
second_order_method = ["Newton", "Hessian_free", "BFGS", "BFGS_momentum"]
total_order_method = ["SGD", "SGD_with_momentum", "SGD_with_nesterov_momentum", "Adagrad", "RMSprop", "Adam", "AdaMax", "NAdam", "Newton_CG", "Hessian_free", "BFGS", "BFGS_momentum"]
style = ["b", "r", "c", "y", "k", "m", "g", "pink", "olivedrab", "deepskyblue", "orange", "darkgrey"]


loss_range_tot = []
x_range_tot = []

valid_acc_tot = []
valid_x_tot = []

plt.figure(figsize=(10, 10))
for i in tqdm(range(8)):
    best_valid_A, x_range, loss_range, valid_x_range, valid_acc_range, best_time, best_epoch = Minimize(X_train, y_train, X_valid, y_valid, method=first_order_method[i])
    plt.plot(valid_x_range, valid_acc_range, style[i])
    loss_range_tot.append(loss_range)
    x_range_tot.append(x_range)
    valid_acc_tot.append(valid_acc_range)
    valid_x_tot.append(valid_x_range)
    test_acc = predict(best_valid_A, X_test, y_test)
    print(f"method {first_order_method[i]}, test acc: {test_acc * 100:.4f}%, best_time: {best_time}s, best_epoch: {best_epoch}")

plt.xlabel("iteration")
plt.ylabel("valid accuracy")
plt.title("Valid accuracy with iteration(first order)")
plt.legend(first_order_method, loc="best")
plt.savefig("logistic_valid_accuracy_first_order.png", dpi=300)

plt.figure(figsize=(10, 10))
for i in tqdm(range(2)):
    best_valid_A, x_range, loss_range, valid_x_range, valid_acc_range, best_time, best_epoch = Minimize(X_train, y_train, X_valid, y_valid, method=second_order_method[i])
    plt.plot(valid_x_range, valid_acc_range, style[i])
    loss_range_tot.append(loss_range)
    x_range_tot.append(x_range)
    valid_acc_tot.append(valid_acc_range)
    valid_x_tot.append(valid_x_range)
    test_acc = predict(best_valid_A, X_test, y_test)
    print(f"method {second_order_method[i]}, test acc: {test_acc * 100:.4f}%, best_time: {best_time}s, best_epoch: {best_epoch}")

plt.xlabel("iteration")
plt.ylabel("valid accuracy")
plt.title("Valid accuracy with iteration(second order)")
plt.legend(second_order_method, loc="best")
plt.savefig("logistic_valid_accuracy_second_order_newton_hessian.png", dpi=300)

plt.figure(figsize=(10, 10))
for i in tqdm(range(2,4)):
    best_valid_A, x_range, loss_range, valid_x_range, valid_acc_range, best_time, best_epoch = Minimize(X_train, y_train, X_valid, y_valid, method=second_order_method[i])
    plt.plot(valid_x_range, valid_acc_range, style[i])
    loss_range_tot.append(loss_range)
    x_range_tot.append(x_range)
    valid_acc_tot.append(valid_acc_range)
    valid_x_tot.append(valid_x_range)
    test_acc = predict(best_valid_A, X_test, y_test)
    print(f"method {second_order_method[i]}, test acc: {test_acc * 100:.4f}%, best_time: {best_time}s, best_epoch: {best_epoch}")

plt.xlabel("iteration")
plt.ylabel("valid accuracy")
plt.title("Valid accuracy with iteration(second order)")
plt.legend(second_order_method[2:4], loc="best")
plt.savefig("logistic_valid_accuracy_second_order_BFGS.png", dpi=300)





