
# Custom numerical integration using the trapezoidal rule (Numba-compatible)
@njit
def trapezoidal_integrate(pdf, a, b, num_points=1000):
    h = (b - a) / num_points
    integral = 0.5 * (pdf(a) + pdf(b))  # Start with endpoints contribution
    for i in range(1, num_points):
        x = a + i * h
        integral += pdf(x)
    return integral * h

# Vectorized CDF computation using Numba
@njit
def vectorized_cdf(pdf, num_points, a=A, b=B, num_points_min = 1000):

    n_inter = int(np.ceil(num_points_min / num_points))
    
    n_tot = num_points * n_inter
    
    cdf_values = np.zeros(num_points)
    
    
    h = (b - a) / n_tot
    integral = 0.
    
    prev_value = 0.5 * pdf(a) * h
    
    x = a
    
    for i in range(num_points):
        
        for j in range(n_inter):
            
            new_value = .5 * pdf(x) * h
            integral += prev_value + new_value
            prev_value = new_value
            x += h
        
        cdf_values[i] = integral
    
    return cdf_values