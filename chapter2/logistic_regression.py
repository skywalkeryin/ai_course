import math
import copy
import numpy as np

class MyLogisticRegression():
    def __init__(self) -> None:
        self.X = None
        self.y = None
        self.w = None
        self.b = None
        pass

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Args:
            X (ndarray (m, n)): Data, m examples with n features 
            y (ndarray (m,))  : Tagets, m examples
        """
        iterations = 10000
        alpha = 0.01
        self.X = X
        self.y = y
        self.w = np.zeros((X.shape[1],))
        self.b = 0.0
        self.w, self.b, cost_history = self.gradient_descent(X, y, self.w, self.b, alpha, iterations)
    
    def predict(self, X):
        """
        Predict using the linear model

        Args:
            X (ndarray (m, n)): Data, m examples with n features 

        Returns:
            ndarray (m,): Predictions
        """
        m = X.shape[0]
        y_pred = np.zeros((m,))
        for i in range(m):
            z_i = X[i]@self.w + self.b
            f_wb_i = self.sigmoid(z_i)
            if f_wb_i >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred
    

    def sigmoid(self, z):
        """
        Compute the sigmoid of z

        Parameters
        ----------
        z : array_like
            A scalar or numpy array of any size.

        Returns
        -------
        g : array_like
            sigmoid(z)
        """
        z = np.clip(z, -500, 500 )           # protect against overflow
        g = 1.0/(1.0+np.exp(-z))

        return g
    
    def compute_cost_logistic(self, X, y, w, b):
        """
        Compute cost

        Args:
            X (ndarray (m, n)): Data, m examples with n features 
            y (ndarray (m,))  : Labels, m examplesSS
            w (ndarray (n,))  : model parameters
            b (scalar)        : model parameter
        Returns:
            cost (scalar)     : cost for logistic regression
        """
        m = X.shape[0]
        cost = 0.0
        for i in range(m):
            z_i = X[i]@w + b
            f_wb_i = self.sigmoid(z_i)
            cost += (-y[i]*np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i))
        cost = cost / m
        return cost

    
    def compute_gradient_logistic(self, X, y, w, b):
        """
        Computes the gradient for linear regression 
    
        Args:
        X (ndarray (m,n): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters  
        b (scalar)      : model parameter
        Returns
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
        dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
        """
        m = X.shape[0] # 样本数量
        n = X.shape[1] # 特征数量 the number of features

        dj_dw = np.zeros((n,))    #(n,)
        dj_db = 0.

        for i in range(m):
            f_wb_i = self.sigmoid(np.dot(X[i], w) + b)  #(n,)(n,)=scalar
            err_i = f_wb_i- y[i]   #scalar
            for j in range(n):
                dj_dw[j] += err_i * X[i, j]  #scalar
            dj_db += err_i 
        
        dj_dw = dj_dw / m   #(n, )
        dj_db = dj_db / m   #scalar
        return dj_db, dj_dw
    
    def gradient_descent(self, X, y, w_in, b_in, alpha, num_iters):
        """
        Performs batch gradient descent

        Args:
            X (ndarray (m, n))  : Data, m examples with n features
            y (ndarray (m, ))   : target values
            w_in (ndarray (n,)) : initial value for the model parameters
            b_in (scalar)       : initial value for the model parameter
            alpha (scalar)      : learning rate
            num_iters (scalar)  : number of iterations

        Returns:
            w (ndarray (n,)) : model parameters after optimization
            b (scalar)       : model parameter after optimization
            J (list)         : cost function history
        """
        cost_history = []
        w = copy.deepcopy(w_in)
        b = b_in

        for i in range(num_iters):
            # calculate the gradient and update the parameters
            dj_db, dj_dw = self.compute_gradient_logistic(X, y, w, b)

            # update the parameters
            w -= alpha * dj_dw
            b -= alpha * dj_db

            # save the cost at each iteration
            if i < 10000000: # prevent resource exhaustion 
                cost_history.append(self.compute_cost_logistic(X, y, w, b))
            
            # Print cost every at intervals 10 times or as many iterations if < 10
            # if i% math.ceil(num_iters / 10) == 0:
            #     print(f"Iteration {i:4d}: Cost {cost_history[-1]}   ")
        
        return w, b, cost_history