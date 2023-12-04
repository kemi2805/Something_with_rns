from scipy.optimize import minimize_scalar, OptimizeResult, brent
import numpy as np


def custom_brent_polynomial(fun, args=(), bracket=None, tol=1e-5, **options):
    history = args[0]

    # First, attempt Brent's method optimization
    result = minimize_scalar(fun, bracket, None, args, 'brent', tol = tol, options = options)
    if isinstance(result, OptimizeResult) and result.success:
        print("Brent Method was a success")
        return result

    # Clean up the history based on the bracket
    history.delete_outside_bounds(bracket[0], bracket[1], last_entry=True)
    
    history.df.to_csv("Hihi.csv")

    # Fetch the last five entries and sort them
    last_five = history.df.iloc[-5:]
    sorted_data = last_five.sort_values(by='rho_c')
    rho_c_vals = sorted_data['rho_c'].tolist()
    M_vals = sorted_data['M'].tolist()
    # Polynomial fitting
    coeffs = np.polyfit(rho_c_vals, M_vals, 3)
    poly = np.poly1d(coeffs)

    # Find critical points of the polynomial
    critical_points = poly.deriv().r
    valid_crit_points = [point for point in critical_points if bracket[0] < point.real < bracket[1]]

    # Determine if the critical point is a local maximum
    rho_TOV, M_TOV = None, None
    if valid_crit_points:
        point = valid_crit_points[0].real  # Taking the first valid critical point
        if poly.deriv(2)(point) < 0:  # Check if it's a local maximum
            rho_TOV = point
            M_TOV = poly(rho_TOV)
        else:
            print("The critical point is not a local maximum.")

    # If we didn't find a maximum, handle the situation (maybe raise an exception or just return)
    if rho_TOV is None:
        return OptimizeResult(x=None, fun=None, success=False, message="No valid maximum found")

    # Evaluate the original function at the maximum
    final_value = fun(rho_TOV, *args)

    return OptimizeResult(x=rho_TOV, fun=final_value, success=True)
