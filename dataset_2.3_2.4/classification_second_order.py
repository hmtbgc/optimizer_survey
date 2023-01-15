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


def Minimize(X, y, epoch_num, X_valid, y_valid, method="Newton_CG"):
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
        loss = loss_function(A, X_train, y_train)
        grad_A = grad_f_A(A, X_train, y_train)
        x_range.append(epoch)
        loss_range.append(loss)

        if (method == "Newton_CG"):
            alpha_ = 0.01
            beta_ = 0.5
            max_iter = 1000
            hessian_matrix = h(A, X_train, y_train)
            d = np.zeros_like(A)
            r = grad_f_A(A, X_train, y_train)
            nr = np.linalg.norm(r)
            CG_tol = min(0.5, nr) * nr
            p = -r
            for i in range(max_iter):
                rr = np.linalg.norm(r) ** 2
                hession_vector_product_p = np.matmul(hessian_matrix, p)
                alpha = rr / np.matmul(p.T, hession_vector_product_p)
                d = d + alpha * p
                r = r + alpha * hession_vector_product_p
                nr1 = np.linalg.norm(r)
                if (nr1 <= CG_tol):
                    break
                beta = nr1 ** 2 / rr
                p = -r + beta * p
            lm = np.matmul(grad_A.T, -d)
            t = 1
            while (loss_function(A, X, y) - loss_function(A + t * d, X, y) < alpha_ * t * lm):
                t = beta_ * t
            A += t * d
            if (lm * lm <= 2 * eps):
                break

        if (method == "Hessian_free"):
            alpha_ = 0.01
            beta_ = 0.5
            hvp = hessian_vector_product(loss_function)
            max_iter = 1000
            d = np.zeros_like(A)
            r = grad_f_A(A, X_train, y_train)
            nr = np.linalg.norm(r)
            CG_tol = min(0.5, nr) * nr
            p = -r
            for i in range(max_iter):
                rr = np.linalg.norm(r) ** 2
                hession_vector_product_p = hvp(A, X_train, y_train, p)
                alpha = rr / np.matmul(p.T, hession_vector_product_p)
                d = d + alpha * p
                r = r + alpha * hession_vector_product_p
                nr1 = np.linalg.norm(r)
                if (nr1 <= CG_tol): 
                    break
                beta = nr1 ** 2 / rr
                p = -r + beta * p
            
            lm = np.matmul(grad_A.T, -d)
            t = 1
            while (loss_function(A, X, y) - loss_function(A + t * d, X, y) < alpha_ * t * lm):
                t = beta_ * t
            A += t * d
            if (lm * lm <= 2 * eps):
                break



        if (method == "BFGS" or method == "BFGS_momentum"):
            alpha_ = 0.01
            beta_ = 0.5
            B = np.eye(A.shape[0])
            if (np.linalg.norm(grad_A) <= eps):
                break
            d = np.matmul(np.linalg.inv(B), -grad_A)
            lm = np.matmul(grad_A.T, -d)

            t = 1
            while (loss_function(A, X, y) - loss_function(A + t * d, X, y) < alpha_ * t * lm):
                t = beta_ * t

            if (method == "BFGS"):
                A += t * d
                s = t * d 
            
            if (method == "BFGS_momentum"):
                momentum = gamma * momentum + t * d
                A += momentum
                s = momentum

            current_grad_A = grad_f_A(A, X_train, y_train)
            yy = current_grad_A - grad_A
            B = B - np.matmul(np.matmul(B, s), np.matmul(s.T, B)) / np.matmul(np.matmul(s.T, B), s) + np.matmul(yy, yy.T) / np.matmul(yy.T, s)
            if (lm * lm <= 2 * eps):
                break
            
        #print(f"epoch {epoch}, train loss:{loss:.4f}")
        if (method == "BFGS"):
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
        else:
            valid_acc = predict(A, X_valid, y_valid)
            valid_acc_range.append(valid_acc)
            valid_x_range.append(epoch)
            if (valid_acc > best_valid_acc):
                best_valid_acc = valid_acc
                best_valid_A = A
                best_time = time.time() - start_time
                best_epoch = epoch


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


second_order_method = ["Newton_CG", "Hessian_free", "BFGS", "BFGS_momentum"]
plt.figure(figsize=(10, 10))
for i in range(2):
    best_valid_A, x_range, loss_range, valid_x_range, valid_acc_range, best_time, best_epoch = Minimize(X_train, y_train, epoch_num, X_valid, y_valid, method=second_order_method[i])
    plt.plot(valid_x_range, valid_acc_range)
    test_acc = predict(best_valid_A, X_test, y_test)
    print(f"method {second_order_method[i]}, test acc: {test_acc * 100:.4f}%, best_time: {best_time}s, best_epoch: {best_epoch}")

plt.xlabel("iteration")
plt.ylabel("valid accuracy")
plt.title("Valid accuracy with iteration(second order)")
plt.legend(second_order_method[:2], loc="best")
plt.savefig("classification_valid_accuracy_second_order_line_search.png", dpi=300)

plt.figure(figsize=(10, 10))
best_valid_A, x_range, loss_range, valid_x_range, valid_acc_range, best_time, best_epoch = Minimize(X_train, y_train, epoch_num, X_valid, y_valid, method=second_order_method[3])
plt.plot(valid_x_range, valid_acc_range)
test_acc = predict(best_valid_A, X_test, y_test)
print(f"method {second_order_method[3]}, test acc: {test_acc * 100:.4f}%, best_time: {best_time}s, best_epoch: {best_epoch}")
plt.xlabel("iteration")
plt.ylabel("valid accuracy")
plt.title("Valid accuracy with iteration(second order)")
plt.legend([second_order_method[3]], loc="best")
plt.savefig("classification_valid_accuracy_second_order_line_search_BFGS_momentum.png", dpi=300)


plt.figure(figsize=(10, 10))
best_valid_A, x_range, loss_range, valid_x_range, valid_acc_range, best_time, best_epoch = Minimize(X_train, y_train, epoch_num, X_valid, y_valid, method=second_order_method[2])
plt.plot(valid_x_range, valid_acc_range)
test_acc = predict(best_valid_A, X_test, y_test)
print(f"method {second_order_method[2]}, test acc: {test_acc * 100:.4f}%, best_time: {best_time}s, best_epoch: {best_epoch}")
plt.xlabel("iteration")
plt.ylabel("valid accuracy")
plt.title("Valid accuracy with iteration(second order)")
plt.legend([second_order_method[2]], loc="best")
plt.savefig("classification_valid_accuracy_second_order_line_search_BFGS.png", dpi=300)









