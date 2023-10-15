import numpy as np

def model(m,n,x0,x00):
    x1 = np.cumsum(x0[0:m])
    z1 = ((x1[:m-1] + x1[1:])/2).reshape((m-1,1))
    x01 = np.cumsum(x00[0:m],axis=0)
    z2 = ((x01[:m-1,:] + x01[1:,:])/2).reshape((m-1,n-1))
    B = np.concatenate((-z1,z2,np.ones(m-1).reshape((m-1,1))),axis=1)
    Y = x0[1:m].reshape(m-1,1)
    u = np.dot(np.linalg.pinv(B),Y)
    return u

def f(b,x,N,u):
    m = []
    n = len(x)
    b = b.flatten()
    for k in range(N):
        A = []
        for i in range(n):
            a = x[i][k]
            A.append(a)
        A = np.array(A)
        A = A.flatten()
        f = np.dot(b,A)
        m.append(f)
        M = np.array(m)+u
    return M

def forecastT(n,m,mm,u,x0,x00):
    u = np.round(u*10000)/10000
    x01 = np.cumsum(x00,axis=0).reshape((m+mm,n-1))#横
    x1_pre = np.zeros(m+mm)
    x0_pre = np.zeros(m+mm)
    x1_pre[0] = x0_pre[0] = x0[0]
    M = f(u[1:-1],x01.T,m+mm,u[-1])
    for i in range(1,m+mm):
        E=0
        for p in range(1,i+1):
            e = (1/2)*((np.exp(-u[0]*(i-p)))*M[p]+(np.exp(-u[0]*(i-p+1))*M[p-1]))
            E = E + e
        x1_pre[i] = x0[0]*np.exp(-u[0]*(i)) + E
    for i in range(1,m+mm):
        x0_pre[i] = x1_pre[i] - x1_pre[i-1]
    PR = np.sqrt((1/m)*np.sum(((x0_pre[0:m]-x0[0:m])/x0[0:m])**2))*100
    PO = np.sqrt((1/m)*np.sum(((x0_pre[m:m+mm]-x0[m:m+mm])/x0[m:m+mm])**2))*100
    return x0_pre,PR,PO


def forecastG(n,m,mm,u,x0,x00):
    u=np.round(u*10000)/10000
    x01=np.cumsum(x00,axis=0).reshape((m+mm,n-1))
    x1_pre = np.zeros(m+mm)
    x0_pre = np.zeros(m+mm)
    x1_pre[0] = x0_pre[0] = x0[0]
    M = f(u[1:n],x01.T,m+mm,u[-1])
    for i in range(1,m+mm):
        E=0
        for p in range(1,i+1):
            e = (np.exp(-u[0]*(i-p+0.5)))*0.5*(M[p-1]+M[p])
            E += e
            x1_pre[i] = np.exp(-u[0]*(i))*x0[0] + E 
    for i in range(1,m+mm):
        x0_pre[i] = x1_pre[i] - x1_pre[i-1]
    PR = np.sqrt((1/m)*np.sum(((x0_pre[0:m]-x0[0:m])/x0[0:m])**2))*100
    PO = np.sqrt((1/m)*np.sum(((x0_pre[m:m+mm]-x0[m:m+mm])/x0[m:m+mm])**2))*100
    return x0_pre,PR,PO


