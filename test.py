import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

def lin_lsq(x,y):
    """
    This function returns the coefficients of the linear regression AND the corresponding linear polyonomial
    of the given input data, x and y, in row or column vector form. It implements the linear version of the Method 
    of Least Squares. It also displays a table where each row contains the x and y coordinates of the data, the 
    linear function of the method of LSQ evaluated at that point and the absolute error of the best-fit linear 
    function and the data. We also display the total squared error at the end.
    """
    n = np.prod(x.shape)
    x = x.reshape(n,1) #if given otherwise, we turn x and y vectors to column vectors
    y = y.reshape(n,1)
    s_x = np.sum(x); s_xx = np.sum(x**2); s_y = np.sum(y); s_xy = np.sum(x*y)
    S = np.array([[s_xx, s_x], [s_x, 4]], float)
    d = np.array([s_xy, s_y], float)
    try:
        S_inv = LA.inv(S) # if LA.det(S)=0 then a LinAlgError exception will be raised
    except LinAlgError:
        print("""With the given data, the system of normal equations of the Method of LSQ, does not have or 
                 has infinite solutions because the coefficient matrix, S, is singular, i.e. it doesnt have an inverse.""")
        return None
    else:
        sol = LA.solve(S,d)
        a = sol[0]; b = sol[1]
        g = np.poly1d([a,b])
        g_x = np.polyval(g,x).reshape(n,1)
        err = y-g_x
        print('|    x    |    y    |   g(x)  |   y-g(x) | \n ----------------------------------------')
        table = np.concatenate((x, y, g_x, err), axis=1)
        for (x_i, y_i, g_xi, err_i) in table:
            print(f'|  {x_i:5.2f}  |  {y_i:5.2f}  |  {g_xi:5.2f}  |  {err_i:6.2f}  |')
        print(f"Also the total squared error is {sum(err**2)[0]:.2f} \n")
        return (a,b), g


x = np.array([1, 3, 5, 7], int)
y = np.array([2.5, 3.5, 6.35, 8.1], float)
(a,b), g = lin_lsq(x,y)
print(f"The coefficients of the linear polyonomial of the Method of LSQ are: a = {a:.2f} and b = {b:.2f}")
print(g)
t = np.linspace(0,10, num=1000)
g_t = g(t)
print(g_t)

# Creating the plot and editing some of its attributes for clarity. We will just copy and paste them for future use.
# We also assign every modification to our plot to a dummy/garbage collecting variable; '_' to prevent unwanted outputs

plt.figure(figsize=(10,5))
plt.scatter(x, y , marker='*', c='red', s=80, label='Our Data')
plt.plot(t, g_t, c='blue', linewidth='2.0', label=r'$g(x)=ax+b$')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid(True)
axes = plt.gca() #gca stands for get current axes
axes.set_xlim([-0.5,10])
axes.set_ylim([-0.5,10])
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18 
plt.legend(loc='best', fontsize=14) #Sets the legend box at the best location
plt.axhline(0, color='black', lw=2)
plt.axvline(0, color='black', lw=2)

plt.show()

