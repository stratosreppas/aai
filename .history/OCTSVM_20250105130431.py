from gurobipy import Model, GRB, quicksum
import numpy as np


class OCTSVM():
    
    def __init__(self, max_depth=2, c1=0.1, c2=0.1, c3=0.1):
        self.D = max_depth
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
    
        # Initialize model
        self.model = Model("OCTSVM")
        
    def fit(self, X, y):
        
        # Helper function to get parent node
        def parent(t):
            return (t-1) // 2
        
        # Add to X the bias term
        X = self.normalize_data(np.c_[X, np.ones(X.shape[0])])
        # squeeze y
        y = np.squeeze(y)
        
        # Constants
        N = len(X)
        T = 2**self.D - 1
        p = len(X[0])
        M = 1e8
        
        # Variables
        delta = self.model.addVar(lb=0, name="delta")
        w = {(t, j): self.model.addVar(lb=-GRB.INFINITY, name=f"w_{t}_{j}") for t in range(T) for j in range(p)}
        # weight_0 = model.addVars(T, lb=-GRB.INFINITY, name="weight_0")
        e = {(i, t): self.model.addVar(lb=0, name=f"e_{i}_{t}") for i in range(N) for t in range(T)}
        beta = {(i, t, j): self.model.addVar(lb=-GRB.INFINITY, name=f"beta_{i}_{t}_{j}") for t in range(T) for i in range(N) for j in range(p)}
        z = {(i, t): self.model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{t}") for i in range(N) for t in range(T)}
        theta = {(i, t): self.model.addVar(vtype=GRB.BINARY, name=f"theta_{i}_{t}") for t in range(T) for i in range(N)}
        xi = {(i, t): self.model.addVar(vtype=GRB.BINARY, name=f"xi_{i}_{t}") for t in range(T) for i in range(N)}
        d = {t: self.model.addVar(vtype=GRB.BINARY, name=f"d_{t}") for t in range(T)}

        # Objective function
        self.model.setObjective(delta + 
                        self.c1 * quicksum(e[i, t] for i in range(N) for t in range(T)) +
                        self.c2 * quicksum(xi[i, t] for i in range(N) for t in range(T)) +
                        self.c3 * quicksum(d[t] for t in range(T)), GRB.MINIMIZE)

        # Constraints
        for t in range(T):
            self.model.addConstr(
                quicksum(w[t, j] * w[t, j] for j in range(p)) <= 2*delta, 
                name=f"nsvm_constraint_{t}"
                )
            
            if(t != 0):
                self.model.addConstr(
                    d[t] <= d[parent(t)],
                    name=f"d_sanity_constraint_{t}"
                )
                
                self.model.addConstr(
                    quicksum(w[t, j] * w[t, j] for j in range(p)) <= M * d[t],
                    name=f"omega_d_constraint_{t}"
                )

        for i in range(N):
            
            for level in range(self.D): 
                self.model.addConstr(
                    quicksum(z[i, t] for t in range(2**level-1, 2**(level+1)-1)) == 1,
                    name = f"sanity_check_1_constraint_{i}_{level}"
                )
                
            for t in range(T):
                
                self.model.addConstr(
                    y[i] * quicksum(w[t, j] * X[i][j] for j in range(p)) - 2 * y[i] * quicksum(beta[i, t, j] * X[i][j] for j in range(p)) >= 1 - e[i, t] - M * (1 - z[i, t]),
                    name=f"RE-SVM_constraint_{i}_{t}"
                )
                
                for j in range(p):
                    self.model.addConstr(
                        beta[i, t, j] == w[t, j]*xi[i, t],
                        name=f"beta_definition_constraint_{i}_{t}_{j}"
                    )
                                
                if t != 0:
                    self.model.addConstr(
                        z[i, t] <= z[i, parent(t)],
                        name=f"z_sanity_constraint_{i}_{t}"
                    )
                
                self.model.addConstr(
                    quicksum(w[t, j] * X[i][j] for j in range(p)) >= -M * (1 - theta[i, t]),
                    name=f"sanity_check_2_contraint_{i}_{t}"
                )
                
                self.model.addConstr(
                    quicksum(w[t, j] * X[i][j] for j in range(p)) <= M * theta[i, t],
                    name=f"sanity_check_3_contraint_{i}_{t}"
                )
                
                if t != 0:
                    if t % 2 == 1:
                        self.model.addConstr(
                            z[i, parent(t)] - z[i, t] <= theta[i, parent(t)],
                            name=f"left_inheritance_constraint_{i}_{t}"
                        )
                    else:
                        self.model.addConstr(
                            z[i, parent(t)] - z[i, t] <= 1 - theta[i, parent(t)],
                            name=f"right_inheritance_constraint_{i}_{t}"
                        )
        
        self.model.Params.TimeLimit = 30
        self.model.Params.OutputFlag = 0
        
        self.model.optimize()

    def predict(self, X_test):
        
        results = []
        
        # Add the bias term
        X_test = self.normalize_data(np.c_[X_test, np.ones(X_test.shape[0])])

        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            
            w = {}
            for v in self.model.getVars():
                # get the weigth vectors
                if v.varName.startswith('w_'):
                    # print(v.varName, v.x)
                    t = int(v.varName.split('_')[1])
                    j = int(v.varName.split('_')[2])
                    if t not in w:
                        w[t]= {}
                    w[t][j] = v.x
            print(w)
            # build the tree
            tree = {}
            node = 0
            for instance in X_test:
                print(instance)
                for _ in range(self.D):
                    logit = 0
                    logit += sum(w[t][j] * instance[j] for j in range(len(instance)))
                if logit < 0:
                    node = 2*node+1
                else:
                    node = 2*node+2
                print(logit)
                results.append(-1) if logit < 0 else results.append(1)
    
        return results

    def normalize_data(self, X):
        for i in range(len(X[0])):
            if max(X[:, i]) - min(X[:, i]) != 0: # avoid division by zero
                X[:, i] = (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))
        return X
        


