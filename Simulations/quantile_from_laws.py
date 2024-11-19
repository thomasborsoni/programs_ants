import numpy as np
from numba import njit
import matplotlib.pyplot as plt


@njit
def vectorized_cdf(pdf, numpoints, a=-np.pi, b=np.pi):
    
    x = a
    h = (b-a) / numpoints
    
    pdf_vect = np.empty(numpoints+1)
    
    for k in range(numpoints+1):
        
        pdf_vect[k] = pdf(x)
        x += h
    
    
    Z = np.cumsum(pdf_vect / 2)
    Y = Z[1:] + Z[:-1]
    
    return Y / Y[-1]


@njit
def inverse_linear(f, n_interp):

    n = len(f) - 1
    inv = np.zeros(n_interp+1)
    inv[-1] = 1.

    x = 1# int(np.floor(f[0])) + 1


    a, b = f[x-1], f[x]

    for y in range(1,n_interp):

        while f[x] < y/n_interp :

            x += 1
            
            if x == n+1:
            
                return inv

        a, b = f[x-1], f[x]
        
        if b <= a :
            
            inv[y] = (x - 1.)/n
            
        else :
        
            inv[y] = ((y/n_interp - a) / (b - a ) + x - 1.)/n

    return inv


@njit
def pdf2_(x):
    return (1 - np.cos(x)) / (2 * np.pi)


@njit
def pdf_direction(x, two_variance=1.0):
    
    return np.exp(-x**2 / two_variance)


@njit
def geom_pdf12(x):
    
    return np.sqrt(pdf_direction(x) * pdf2_(x))



@njit
def geom_quantile_function(pdf1, pdf2, numpoints = 1000, numpoints_inv = 1000):
    
    return inverse_linear(vectorized_cdf(geom_pdf12, numpoints), numpoints_inv)



if __name__ == '__main__':


    # Get quantile functions
    qtf = geom_quantile_function(pdf_direction, pdf_direction)
    
    # Generate points for plotting
    x_values = np.linspace(-np.pi, np.pi, 1001, endpoint=False)
    
    
    # Plot the quantile functions
    plt.figure(figsize=(10, 6))
    #plt.plot(x_values, qtf)
    #plt.plot(x_values, vectorized_cdf(geom_pdf12, 1001))
    plt.plot(x_values, pdf_direction(x_values))
    #plt.plot(x_values, quantile2_values, label="Quantile of PDF2: (1 - cos(x)) / (2π)", color="green")
    plt.title("Quantile Functions of PDF1 and PDF2")
    plt.xlabel("Probability (p)")
    plt.ylabel("Quantile (x)")
    plt.legend()
    plt.grid(True)
    plt.show()
