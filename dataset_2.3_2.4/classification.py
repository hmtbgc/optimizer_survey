from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import autograd.numpy as np
from autograd import grad, hessian, hessian_vector_product
from tqdm import tqdm
from matplotlib import pyplot as plt
import time

np.random.seed(1234)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=1234)
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)

train_size = X_train.shape[0]
valid_size = X_valid.shape[0]
test_size = X_test.shape[0]

input_size = 4
output_size = 3

# A should be a vector
def linear(A, x):
    out = []
    for i, j in enumerate(range(0, A.shape[0], input_size)):
        out.append(np.matmul(x, A[j:j+input_size]))
    return out

def loss_function(A, x, label):
    out = linear(A, x)
    loss = 0
    t = np.zeros_like(out[0])
    for i in range(len(out)):
        t += np.exp(out[i])
    loss += np.sum(np.log(t))
    for i in range(label.shape[0]):
        index = label[i]
        loss -= out[index][i]
    return loss

    

epoch_num = 5000
grad_f_A = grad(loss_function, 0)
h = hessian(loss_function)

# lr = 0.1  
# for adagrad, lr should be large enough since lr decreases with iteration

# lr = 0.001 
# for RMSprop

lr = 0.0005
gamma = 0.9
beta_1 = 0.9
beta_2 = 0.999

nesterov = False
eplison = 1e-10
eps = 1e-6



def Minimize(X, y, epoch_num, X_valid, y_valid, method="SGD"):
    A = np.zeros(input_size * output_size)
    best_valid_acc = 0.0
    best_valid_A = np.zeros_like(A)
    best_epoch = 0
    best_time = 0
    momentum = np.zeros_like(A)
    v = np.zeros_like(A)       
    valid_x_range = []
    valid_acc_range = []      
    loss_range = []
    x_range = []            
    start_time = time.time()                            
    for epoch in tqdm(range(epoch_num)):
        t = epoch + 1
        loss = loss_function(A, X_train, y_train)
        grad_A = grad_f_A(A, X_train, y_train)
        x_range.append(epoch)
        loss_range.append(loss)
        if (method == "SGD"):
            A -= lr * grad_A 
        if (method == "SGD_with_momentum"):
            momentum = gamma * momentum + lr * grad_A
            A -= momentum
        if (method == "SGD_with_nesterov_momentum"):
            grad_A_ = grad_f_A(A - gamma * momentum, X_train, y_train)
            momentum = gamma * momentum + lr * grad_A_
            A -= momentum
        if (method == "Adagrad"):
            lr_adagrad = 0.1
            v += grad_A ** 2
            A -= lr_adagrad / np.sqrt(v + eps) * grad_A
        if (method == "RMSprop"):
            lr_rmsprop = 0.01
            v = gamma * v + (1 - gamma) * (grad_A ** 2)
            A -= lr_rmsprop / np.sqrt(v + eplison) * grad_A
        if (method == "Adam"):
            lr_adam = 0.01
            momentum = beta_1 * momentum + (1 - beta_1) * grad_A
            v = beta_2 * v + (1 - beta_2) * (grad_A ** 2)
            momentum_hat = momentum / (1.0 - np.power(beta_1, t))
            v_hair = v / (1.0 - np.power(beta_2, t))
            A -= lr_adam * momentum_hat / (np.sqrt(v_hair) + eplison)
        if (method == "AdaMax"):
            momentum = beta_1 * momentum + (1 - beta_1) * grad_A
            momentum_hat = momentum / (1.0 - np.power(beta_1, t))
            v = np.maximum(beta_2 * v, np.abs(grad_A))
            A -= lr * momentum_hat / (v + eplison)
        if (method == "NAdam"):
            momentum = beta_1 * momentum + (1 - beta_1) * grad_A
            momentum_hat = momentum / (1.0 - np.power(beta_1, t))
            momentum_bar = (beta_1 * momentum_hat + (1 - beta_1) * grad_A) / (1.0 - np.power(beta_1, t))
            v = beta_2 * v + (1 - beta_2) * (grad_A ** 2)
            v_hair = v / (1.0 - np.power(beta_2, t))
            A -= lr * momentum_bar / (np.sqrt(v_hair) + eplison)

        if (method == "Newton_CG"):
            max_iter = 1000
            hessian_matrix = h(A, X_train, y_train)
            g = np.zeros_like(A)
            r = grad_f_A(A, X_train, y_train)
            nr = np.linalg.norm(r)
            CG_tol = min(0.5, nr) * nr
            p = -r
            for i in range(max_iter):
                rr = np.linalg.norm(r) ** 2
                hession_vector_product_p = np.matmul(hessian_matrix, p)
                alpha = rr / np.matmul(p.T, hession_vector_product_p)
                g = g + alpha * p
                r = r + alpha * hession_vector_product_p
                nr1 = np.linalg.norm(r)
                if (nr1 <= CG_tol):
                    break
                beta = nr1 ** 2 / rr
                p = -r + beta * p
            A += lr * g

        if (method == "Hessian_free"):
            hvp = hessian_vector_product(loss_function)
            max_iter = 1000
            g = np.zeros_like(A)
            r = grad_f_A(A, X_train, y_train)
            nr = np.linalg.norm(r)
            CG_tol = min(0.5, nr) * nr
            p = -r
            for i in range(max_iter):
                rr = np.linalg.norm(r) ** 2
                hession_vector_product_p = hvp(A, X_train, y_train, p)
                alpha = rr / np.matmul(p.T, hession_vector_product_p)
                g = g + alpha * p
                r = r + alpha * hession_vector_product_p
                nr1 = np.linalg.norm(r)
                if (nr1 <= CG_tol): 
                    break
                beta = nr1 ** 2 / rr
                p = -r + beta * p
            A += lr * g

        if (method == "BFGS" or method == "BFGS_momentum"):
            B = np.eye(A.shape[0])
            if (np.linalg.norm(grad_A) <= eps):
                break
            d = np.matmul(np.linalg.inv(B), -grad_A)
            lm = np.matmul(grad_A.T, -d)

            if (method == "BFGS"):
                A += lr * d
                s = lr * d 
            
            if (method == "BFGS_momentum"):
                momentum = gamma * momentum + lr * d
                A += momentum
                s = momentum

            current_grad_A = grad_f_A(A, X_train, y_train)
            yy = current_grad_A - grad_A
            B = B - np.matmul(np.matmul(B, s), np.matmul(s.T, B)) / np.matmul(np.matmul(s.T, B), s) + np.matmul(yy, yy.T) / np.matmul(yy.T, s)
            
        #print(f"epoch {epoch}, train loss:{loss:.4f}")
        if (epoch % 100 == 0):
            valid_acc = predict(A, X_valid, y_valid)
            valid_acc_range.append(valid_acc)
            valid_x_range.append(epoch)
            if (valid_acc > best_valid_acc):
                best_valid_acc = valid_acc
                best_valid_A = A
                best_time = time.time() - start_time
                best_epoch = epoch
            #print(f"valid acc:{valid_acc * 100:.4f}%")

    return best_valid_A, x_range, loss_range, valid_x_range, valid_acc_range, best_time, best_epoch


def predict(A, X, y):
    out = linear(A, X)
    result = 0
    for i in range(y.shape[0]):
        true_label = y[i]
        max_score = 0.0
        pred_label = -1
        for j in range(len(out)):
            if (out[j][i] > max_score):
                max_score = out[j][i]
                pred_label = j
        result += (true_label == pred_label)
    acc = result / y.shape[0]
    return acc


first_order_method = ["SGD", "SGD_with_momentum", "SGD_with_nesterov_momentum", "Adagrad", "RMSprop", "Adam", "AdaMax", "NAdam"]
second_order_method = ["Newton_CG", "Hessian_free", "BFGS", "BFGS_momentum"]
total_order_method = ["SGD", "SGD_with_momentum", "SGD_with_nesterov_momentum", "Adagrad", "RMSprop", "Adam", "AdaMax", "NAdam", "Newton_CG", "Hessian_free", "BFGS", "BFGS_momentum"]
style = ["b", "r", "c", "y", "k", "m", "g", "pink", "olivedrab", "deepskyblue", "orange", "darkgrey"]

loss_range_tot = []
x_range_tot = []

valid_acc_tot = []
valid_x_tot = []

# test for first order method
plt.figure(figsize=(10, 10))
for i in tqdm(range(8)):
    best_valid_A, x_range, loss_range, valid_x_range, valid_acc_range, best_time, best_epoch = Minimize(X_train, y_train, epoch_num, X_valid, y_valid, method=first_order_method[i])
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
plt.savefig("classification_valid_accuracy_first_order.png", dpi=300)

plt.figure(figsize=(10, 10))
for i in tqdm(range(8)):
    plt.plot(x_range_tot[i], loss_range_tot[i], style[i])
plt.xlabel("iteration")
plt.ylabel("train loss")
plt.title("Train loss with iteration(first order)")
plt.legend(first_order_method, loc="best")
plt.savefig("classification_train_loss_first_order.png", dpi=300)

# test for second order method

loss_range_tot = []
x_range_tot = []
plt.figure(figsize=(10, 10))
for i in tqdm(range(4)):
    best_valid_A, x_range, loss_range, valid_x_range, valid_acc_range, best_time, best_epoch = Minimize(X_train, y_train, epoch_num, X_valid, y_valid, method=second_order_method[i])
    plt.plot(valid_x_range, valid_acc_range, style[8+i])
    x_range_tot.append(x_range)
    loss_range_tot.append(loss_range)
    valid_acc_tot.append(valid_acc_range)
    valid_x_tot.append(valid_x_range)
    test_acc = predict(best_valid_A, X_test, y_test)
    print(f"method {second_order_method[i]}, test acc: {test_acc * 100:.4f}%, best_time: {best_time}s, best_epoch: {best_epoch}")

plt.xlabel("iteration")
plt.ylabel("valid accuracy")
plt.title("Valid accuracy with iteration(second order)")
plt.legend(second_order_method, loc="best")
plt.savefig("classification_valid_accuracy_second_order.png", dpi=300)

plt.figure(figsize=(10, 10))
for i in tqdm(range(4)):
    plt.plot(x_range_tot[i], loss_range_tot[i], style[8+i])
plt.xlabel("iteration")
plt.ylabel("train loss")
plt.title("Train loss with iteration(second order)")
plt.legend(second_order_method, loc="best")
plt.savefig("classification_train_loss_second_order.png", dpi=300)

plt.figure(figsize=(10, 10))
for i in tqdm(range(8 + 4)):
    plt.plot(valid_x_tot[i], valid_acc_tot[i], style[i])

plt.xlabel("iteration")
plt.ylabel("valid accuracy")
plt.title("Valid accuracy with iteration")
plt.legend(total_order_method, loc="best")
plt.savefig("classification_valid_accuracy_second_order.png", dpi=300)





    







