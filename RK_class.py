import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math as m
import types

class RK_method:
    def __init__(self, A, b, c, n=None):
        self.A = A
        self.b = b
        self.c = c
        self.n = n if n else len(A)
        self.s=len(A)
        self.E=1e-15
        self.method_param_check()
        self.b_teta=None
        
    def method_param_check(self):
        assert self.n<=self.s, "incorrect method data. Check n"
        assert self.s==len(self.b) and self.s==len(self.c), "incorrect method data. Check the dimensions of vectors and matrices"
        assert abs(sum(self.b)-1.0)<=self.E, f"incorrect method data. sum(b)={sum(self.b)}!=1"

        for i in range(self.s):
            assert len(self.A[i])==self.s, "incorrect method data. Check the dimensions of vectors and matrices"
            assert abs(sum(self.A[i])-self.c[i])<=self.E, f"incorrect method data. sum(A[{i}])!=c[{i}]."
        for i in range(self.s):
            for j in range(i, self.s):
                assert self.A[i][j]==0, f"incorrect method data.A[{i}][{j}]={self.A[i][j]}!=0.The explicit method implies a triangular matrix A"
    
    def set_b_teta(self, b_teta):
        for b in b_teta:
            assert isinstance(b(1), (int, float)), "b_theta shuld be an array of functions"
        self.b_teta=b_teta
    
    def b_teta_calc(self):
        teta = sp.Symbol('teta')
        b_roots=[sp.Symbol(f'b{i}') for i in range(len(self.b))]
        sysEq=[sum(b_roots)-teta, 
            
                sum([b_roots[i]*self.c[i] for i in range(1,self.s)])-teta**2/2, 

                sum([b_roots[i]*self.c[i]**2 for i in range(1,self.s)])-teta**3/3,
                sum([b_roots[i]*sum([self.A[i][j]*self.c[j] for j in range(i)]) for i in range(self.s)])-teta**3/6,

                sum([b_roots[i]*self.c[i]**3 for i in range(1,self.s)])-teta**4/4,
                sum([b_roots[i]*self.c[i]*sum([self.A[i][j]*self.c[j] for j in range(i)]) for i in range(self.s)])-teta**4/8,
                sum([b_roots[i]*sum([self.A[i][j]*self.c[j]**2 for j in range(i)]) for i in range(self.s)])-teta**4/12,
                sum([b_roots[i]*sum([self.A[i][j]*sum([self.A[j][k]*self.c[k] for k in range(j)]) for j in range(i)]) for i in range(self.s)])-teta**4/24,
                ]
        
        assert self.n<=4, 'cant calculate for n>4. Use set_b_teta(self, b_teta)'
        if(self.n==1):
            sysEq=sysEq[0]
        if(self.n==2):
            sysEq=sysEq[0:2]
        if(self.n==3):
            sysEq=sysEq[0:4]
            for j in range(1,3):
                sysEq[-1]=sysEq[-1].subs(teta,1)
                for i in range(self.n):
                    sysEq[-1]=sysEq[-1].subs(b_roots[i],self.b[i])
                if(abs(sysEq[-1])<self.E):sysEq=sysEq[0:-1]
                else: print('не удовлетворяет необходимому условию порядка', sysEq[-1])
        if(self.n==4):
            sysEq=sysEq[0:4]
        sysEq=[sp.nsimplify(eq) for eq in sysEq]
        b_teta = sp.solve(sysEq)
        b_teta=[sp.nsimplify(sp.expand(b_teta[0][root])) for root in b_teta[0]]
        while(len(b_teta)<len(self.b)):
            b_missing=self.b[len(b_teta)]*teta**(self.n-1)
            b_teta=[b_.subs(b_roots[len(b_teta)],b_missing) for b_ in b_teta]
            b_teta.append(b_missing)
        self.b_teta=[eval("lambda teta:" + str(sp.nsimplify(root))) for root in b_teta]

    #solution for DDE and SDDE 
    def get_solution_DDE(self, x0:float, x1:float, y0, f, h:float, tau):
        assert callable(f), "f(x,y)-eq for sol"
        assert x0+h<x1, "x0-start of sol, x1-end, h-step. It shouldnt be x0+h>x1"
        assert callable(y0), "f(x,y)-eq for sol"
        if(not self.b_teta):
            print('need some time for find b_teta')
            self.b_teta_calc()
            print('done it. Starting to calculate solution')
   
        if(not isinstance(tau,(list,np.ndarray))):
            tau=[tau]

        X=[]
        K_history=[] 
        if isinstance(y0(x0),(list,np.ndarray)):
            Y=[np.array(y0(x0))]
            def K(x,y,f, h, tau):
                K_i=[]
                for i in range(self.s):
                    K_i.append(np.array(f(x+self.c[i]*h, y+h*sum([self.A[i][j]*K_i[j] for j in range(i)]), *[y_tau(x+self.c[i]*h, tau_) for tau_ in tau])))
                return K_i
        else:
            if isinstance(y0(x0),(int,float)):
                Y=[y0(x0)]
                def K(x,y,f, h, tau):
                    K_i=[]
                    for i in range(self.s):
                        K_i.append(f(x+self.c[i]*h, y+h*sum([self.A[i][j]*K_i[j] for j in range(i)]), *[y_tau(x+self.c[i]*h, tau_) for tau_ in tau]))
                    return K_i
            else: assert False, "y0 shold be list or np.ndarray if you are trying to get system solution. y0 shold be int or float if you are trying to get eq solution."

        #поиск пары чисел в упорядоченном массиве, между которыми мог бы заключен x
        def BS(x):
            assert x<=X[-1], "cant solve eq with such dellay."
            low = 0
            high = len(X) - 1
            while low <= high:
                mid = (low + high) // 2
                midVal = X[mid]
                if mid+1<len(X) and (midVal < x and X[mid+1] > x):
                    return midVal, X[mid+1], mid+1
                if mid-1>=0 and (midVal > x and X[mid-1] < x):
                    return X[mid-1], midVal, mid
                if midVal > x:
                    high = mid - 1
                else:
                    low = mid + 1
            print('not found :(')
            print(X, x)
            return

        def y_tau(x, tau):
            x-=tau
            if x<x0:
                return y0(x) #предыстория (за рамками интервала)
            if x in X:
                return Y[X.index(x)] #искомый y уже был посчитан
            x_m_1, x_m, m = BS(x)
            tet = (x-x_m_1)/(x_m-x_m_1)
            return Y[m-1]+h*sum([K_history[m-1][i]*self.b_teta[i](tet) for i in range(self.s)]) #рассчет y

        
        for x in np.arange(x0, x1, h):
            if(x+h>x1):
                h=abs(x1-x)
            X.append(x)
            K_history.append(K(x,Y[-1],f, h, tau))
            Y.append(Y[-1]+h*sum([self.b[i]*K_history[-1][i] for i in range(self.s)]))
        X.append(x1)
         
        return(X, Y)

    #solution for ODE and SODE 
    def get_solution(self, x0:float, x1:float, y0, f, h:float):
        assert callable(f), "f(x,y)-eq for sol"
        assert x0+h<x1, "x0-start of sol, x1-end, h-step. It shouldnt be x0+h>x1"
        if isinstance(y0,(list,np.ndarray)):
            y0=np.array(y0)
            def K(x,y,f, h):
                K_i=[]
                for i in range(self.s):
                    K_i.append(np.array(f(x+self.c[i]*h, y+h*sum([self.A[i][j]*K_i[j] for j in range(i)]))))
                return K_i
        else:
            if isinstance(y0,(int,float)):
                def K(x,y,f, h):
                    K_i=[]
                    for i in range(self.s):
                        K_i.append(f(x+self.c[i]*h, y+h*sum([self.A[i][j]*K_i[j] for j in range(i)])))
                    return K_i
            else: assert False, "y0 shold be list or np.ndarray if you are trying to get system solution. y0 shold be int or float if you are trying to get eq solution."

        Y=[y0]
        X=[]
        for x in np.arange(x0, x1, h):
            if(x+h>x1):
                h=abs(x1-x)
            X.append(x)
            K_i=K(x,Y[-1],f, h)
            Y.append(Y[-1]+h*sum([self.b[i]*K_i[i] for i in range(self.s)]))
        X.append(x1)
         
        return(X, Y)
    
    #Conversation
    def get_Graph_Conversation(self, x0:float, x1:float, y0, f, y1_true, tau=0, n=None):
        if callable(y1_true):
            y1_true=y1_true(x1)
        n = n if n else self.n
        Norm = []
        H = []
        for k in range(6, 11):
            h = 1 / (2 ** k)
            H.append(np.log10(h))
            if callable(y0):
                X,Y = self.get_solution_DDE(x0, x1, y0, f, h, tau)
            else:
                X,Y = self.get_solution(x0, x1, y0, f, h)

            if isinstance(y1_true,(list,np.ndarray)):
                Norm.append(np.log10(np.linalg.norm([Y[-1][i]-y1_true[i] for i in range(self.s)])))
            else:
                Norm.append(np.log10(abs(Y[-1]-y1_true)))
        print('H:', H)
        print('Norm:', Norm)
        x = np.linspace(0,0.06,100)
        y = (Norm[0]) + n*(H-H[0])
        fig, ax = plt.subplots()
        ax.plot(H, y, 'r')
        ax.plot(H, Norm, color='b', linestyle='--')
        
        plt.xlabel("log10(Длина шага)")
        plt.ylabel("log10(Норма погрешности)")
        return H, Norm

    #Solution
    def get_Graph_Solution(self, X, Y, y_true=None):
        fig, ax = plt.subplots()
        if(isinstance(Y[0],(list,np.ndarray))):
            for i in range(len(Y[0])):
                ax.plot(X, [y[i] for y in Y], color='b')
        else:
            ax.plot(X, Y, 'b')
        
        plt.xlabel("x")
        plt.ylabel("y")
        if(y_true):
            assert callable(y_true), "y_true(x)-analitical sol"
            if(isinstance(y_true(X[0]),(list,np.ndarray))):
                for i in range(len(y_true(X[0]))):
                    ax.plot(X, [y_true(x)[i] for x in X], color='r', linestyle='--')
            else:
                ax.plot(X, [y_true(x) for x in X], color='r', linestyle='--')

class VIDE_RK_method(RK_method):
    def __init__(self, e, w, d, A, b, c, n=None):
        super().__init__(A, b, c, n)
        self.e=e
        self.w=w
        self.d=d
        self.method_param_check_()

    def method_param_check_(self):
        for i in range(self.s-1):
            assert self.d[i]>=self.c[i],f'{self.d[i]},< {self.c[i]}'
        assert len(self.e)==len(self.w),'w and e are vectors for counting intergration tail. they should have the same length.'
        assert len(self.d)==self.s,'d should be the same length as others method vectors = s.'
        
    def get_solution_VIDE(self, x0:float, x1:float, y0, f, K, h:float, limit=None):
        limit=limit if limit else lambda x:x0
        assert callable(f), "f(x,y,F)-eq for sol"
        assert callable(K), "K(x,s, y(s))-integration eq for sol"
        assert callable(y0), "y0(t) should retern past"
        assert x0+h<x1, "x0-start of sol, x1-end, h-step. It shouldnt be x0+h>x1"
        if(not self.b_teta):
            print('need some time for find b_teta')
            self.b_teta_calc()
            print('done it. Starting to calculate solution')

        def binarySearch(x, arr):
            mid = len(arr) // 2
            low = 0
            high = len(arr) - 1

            while arr[mid] != x and low <= high:
                if x > arr[mid]:
                    low = mid + 1
                else:
                    high = mid - 1
                mid = (low + high) // 2

            return mid
        def new_y(i, h, teta):
            if(teta==0):return Y[i]
            if(teta==1 and len(Y)-1!=i):return Y[i+1]
            return Y[i]+h*sum([F[i][j]*self.b_teta[j](teta) for j in range(4)])
        
        def integration_K(t):
            res = 0
            start = binarySearch(limit(t), X)
            end = binarySearch(t, X)
            if start<0:
                if end<0:
                    for i in range(limit(t), t, h):
                        res+=h*sum([self.w[j]*K(t, i+h*self.e[j],y0(i+h*self.e[j])) for j in range(len(self.w))])
                    return res
                for i in range(limit(t), x0, h):
                    res+=h*sum([self.w[j]*K(t, i+h*self.e[j],y0(i+h*self.e[j])) for j in range(len(self.w))])
                start+=1
            elif limit(t)!=X[start]:
                h_ = X[start + 1] - limit(t)
                res+=h_*sum([self.w[j]*K(t, limit(t)+h_*self.e[j],new_y(start, h_, self.e[j])) for j in range(len(self.w))])
                start+=1
            
            for i in range(start, end):
                res+=h*sum([self.w[j]*K(t, X[i]+h*self.e[j],new_y(i, h, self.e[j])) for j in range(len(self.w))])
                if(m.isnan(res)):
                    print(sum([self.w[j]*K(t, X[i]+h*self.e[j],new_y(i, h, self.e[j])) for j in range(len(self.w))]))
            
            return res
        
        def Y_Z_f(h):
            Y_=[]
            Z_=[]
            F_=[integration_K(X[-1]+self.c[i]*h) for i in range(self.s)]
            for i in range(self.s):
                Y_.append(Y[-1]+sum([self.A[i][j]*f(X[-1]+self.c[j]*h,Y_[j],F_[j]+Z_[j]) for j in range(i)]))
                Z_.append(h*sum([self.A[i][j]*K(X[-1]+self.d[j]*h,X[-1]+self.c[j]*h,Y_[j]) for j in range(i)]))
            f_=[f(X[-1]+self.c[i]*h,Y_[i],F_[i]+Z_[i]) for i in range(self.s)]
            return(f_)

        X=[x0]
        Y=[y0(x0)]
        F=[]
        while X[-1]+h<=x1:
            F.append(Y_Z_f(h))
            Y.append(new_y(len(F)-1, h, 1))
            X.append(X[-1]+h)
        if X[-1]!=x1:
            F.append(Y_Z_f(x1-X[-1]))
            Y.append(new_y(len(F)-1, h, 1))
            X.append(x1)
        return(X,Y) 
  