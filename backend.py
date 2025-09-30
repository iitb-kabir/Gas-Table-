import math
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Calculation Functions
# -----------------------------

def isentropic_calculate(gamma, input_type, input_value, is_sub=None):
    # Helper functions for forward calculations from Mach number
    def t_t0_from_m(m, g):
        return 1 / (1 + 0.5 * (g - 1) * m**2)

    def p_p0_from_t_t0(t_t0, g):
        return t_t0 ** (g / (g - 1))

    def rho_rho0_from_t_t0(t_t0, g):
        return t_t0 ** (1 / (g - 1))

    def mach_angle(m):
        if m < 1:
            return None
        return math.degrees(math.asin(1 / m))

    def pm_angle(m, g):
        if m <= 1:
            return 0
        a = math.sqrt((g + 1) / (g - 1))
        b = math.sqrt((g - 1) / (g + 1) * (m**2 - 1))
        c = math.sqrt(m**2 - 1)
        return math.degrees(a * math.atan(b) - math.atan(c))

    def area_ratio(m, g):
        if m == 0:
            return float("inf")
        temp = 1 + 0.5 * (g - 1) * m**2
        return (1 / m) * ((2 / (g + 1)) * temp) ** (0.5 * (g + 1) / (g - 1))

    # Critical condition ratios (where M=1)
    def t_t0_star(g):
        return 2 / (g + 1)

    def p_p0_star(g):
        return (2 / (g + 1)) ** (g / (g - 1))

    def rho_rho0_star(g):
        return (2 / (g + 1)) ** (1 / (g - 1))

    # Numerical solver for implicit equations
    def bisection_solve(func, target, low, high, increasing=True, tol=1e-8):
        while high - low > tol:
            mid = (low + high) / 2
            val = func(mid)
            if increasing:
                if val < target:
                    low = mid
                else:
                    high = mid
            else:
                if val < target:
                    high = mid
                else:
                    low = mid
        return (low + high) / 2

    # Find Mach number (M) from the given input
    m = 0
    if input_type == "mach":
        m = input_value
    elif input_type == "mach_angle":
        m = 1 / math.sin(math.radians(input_value))
    elif input_type == "pm_angle":
        def pm_func(m_val):
            return pm_angle(m_val, gamma)
        m = bisection_solve(pm_func, input_value, 1.0001, 80, increasing=True)
    elif input_type == "p_p0":
        t_t0_val = input_value ** ((gamma - 1) / gamma)
        m = math.sqrt(2 / (gamma - 1) * (1 / t_t0_val - 1))
    elif input_type == "rho_rho0":
        t_t0_val = input_value ** (gamma - 1)
        m = math.sqrt(2 / (gamma - 1) * (1 / t_t0_val - 1))
    elif input_type == "t_t0":
        t_t0_val = input_value
        m = math.sqrt(2 / (gamma - 1) * (1 / t_t0_val - 1))
    elif input_type == "a_a_star":
        if is_sub is None:
            raise ValueError("For A/A* input, must specify if flow is subsonic or supersonic.")
        if input_value < 1:
            raise ValueError("A/A* must be >= 1 (minimum at M=1).")
        def area_func(m_val):
            return area_ratio(m_val, gamma)
        if is_sub:
            m = bisection_solve(area_func, input_value, 0.0001, 0.9999, increasing=False)
        else:
            m = bisection_solve(area_func, input_value, 1.0001, 80, increasing=True)

    # Compute all output properties from M
    t_t0 = t_t0_from_m(m, gamma)
    p_p0 = p_p0_from_t_t0(t_t0, gamma)
    rho_rho0 = rho_rho0_from_t_t0(t_t0, gamma)
    mu = mach_angle(m)
    nu = pm_angle(m, gamma)
    a_a_star = area_ratio(m, gamma)

    t_t_star = t_t0 / t_t0_star(gamma)
    p_p_star = p_p0 / p_p0_star(gamma)
    rho_rho_star = rho_rho0 / rho_rho0_star(gamma)

    return {
        "Mach": m, "Mach angle": mu, "P-M angle": nu,
        "p/p0": p_p0, "rho/rho0": rho_rho0, "T/T0": t_t0,
        "p/p*": p_p_star, "rho/rho*": rho_rho_star, "T/T*": t_t_star,
        "A/A*": a_a_star,
    }

def normal_shock_calculate(gamma, input_type, input_value):
    # Find upstream Mach number (M1) from the given input
    def m1_from_m2(m2, g):
        if m2 >= 1: raise ValueError("M2 must be subsonic (< 1).")
        num = 2 + (g - 1) * m2**2
        den = 2 * g * m2**2 - (g - 1)
        if den <= 0: raise ValueError("Invalid M2 for a physical solution.")
        return math.sqrt(num / den)

    def m1_from_p2_p1(p_ratio, g):
        if p_ratio <= 1: raise ValueError("p2/p1 must be > 1.")
        return math.sqrt(((g + 1) * p_ratio + (g - 1)) / (2 * g))

    def m1_from_rho2_rho1(rho_ratio, g):
        limit = (g + 1) / (g - 1)
        if not 1 < rho_ratio < limit:
            raise ValueError(f"rho2/rho1 must be between 1 and {limit:.4f}.")
        num = 2 * rho_ratio
        den = (g + 1) - (g - 1) * rho_ratio
        return math.sqrt(num / den)

    def bisection_solve(func, target, low, high, increasing=True, tol=1e-8):
        while high - low > tol:
            mid = (low + high) / 2
            val = func(mid)
            if increasing:
                if val < target:
                    low = mid
                else:
                    high = mid
            else:
                if val < target:
                    high = mid
                else:
                    low = mid
        return (low + high) / 2

    m1 = 0
    if input_type == "m1":
        m1 = input_value
        if m1 <= 1: raise ValueError("M1 must be supersonic (> 1).")
    elif input_type == "m2":
        m1 = m1_from_m2(input_value, gamma)
    elif input_type == "p2_p1":
        m1 = m1_from_p2_p1(input_value, gamma)
    elif input_type == "rho2_rho1":
        m1 = m1_from_rho2_rho1(input_value, gamma)
    elif input_type == "t2_t1":
        if input_value <= 1: raise ValueError("T2/T1 must be > 1.")
        def t_ratio_func(m_val):
            p_r = (2 * gamma * m_val**2 - (gamma - 1)) / (gamma + 1)
            rho_r = ((gamma + 1) * m_val**2) / ((gamma - 1) * m_val**2 + 2)
            return p_r / rho_r
        m1 = bisection_solve(t_ratio_func, input_value, 1.0001, 80, increasing=True)
    elif input_type == "po2_po1":
        if input_value >= 1: raise ValueError("Po2/Po1 must be < 1.")
        def po_ratio_func(m_val):
            t1 = ((gamma + 1) * m_val**2) / ((gamma - 1) * m_val**2 + 2)
            t2 = (gamma + 1) / (2 * gamma * m_val**2 - (gamma - 1))
            return t1**(gamma / (gamma - 1)) * t2**(1 / (gamma - 1))
        m1 = bisection_solve(po_ratio_func, input_value, 1.0001, 80, increasing=False)
    elif input_type == "p1_po2":
        p_star_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        if input_value > p_star_ratio:
            raise ValueError(f"p1/Po2 cannot be greater than p*/po = {p_star_ratio:.4f}")
        def p1_po2_func(m_val):
            term1 = ((gamma + 1) * m_val**2 / 2)**(gamma / (gamma - 1))
            term2 = ((gamma + 1) / (2 * gamma * m_val**2 - (gamma - 1)))**(1 / (gamma - 1))
            po2_p1 = term1 * term2
            return 1 / po2_p1
        m1 = bisection_solve(p1_po2_func, input_value, 1.0001, 80, increasing=False)
    else:
        raise ValueError(f"Invalid input type for Normal Shock: {input_type}")

    # Compute all properties from M1
    m2 = math.sqrt((2 + (gamma - 1) * m1**2) / (2 * gamma * m1**2 - (gamma - 1)))
    p2_p1 = (2 * gamma * m1**2 - (gamma - 1)) / (gamma + 1)
    rho2_rho1 = ((gamma + 1) * m1**2) / ((gamma - 1) * m1**2 + 2)
    t2_t1 = p2_p1 / rho2_rho1
    po2_po1 = (((gamma + 1) * m1**2) / ((gamma - 1) * m1**2 + 2)) ** (gamma / (gamma - 1)) * \
              ((gamma + 1) / (2 * gamma * m1**2 - (gamma - 1))) ** (1 / (gamma - 1))
    
    po2_p1_val = (1 + (gamma-1)/2 * m2**2)**(gamma/(gamma-1)) * p2_p1

    return {
        "M1": m1, "M2": m2, "p2/p1": p2_p1, "rho2/rho1": rho2_rho1,
        "T2/T1": t2_t1, "Po2/Po1": po2_po1, "Po2/p1": po2_p1_val, "p1/Po2": 1/po2_p1_val
    }

def oblique_shock_calculate(gamma, m1, input_type, input_value, is_weak=None):
    if m1 <= 1: raise ValueError("M1 must be supersonic (> 1).")

    def theta_from_beta(beta_rad, m_val, g):
        if m_val * math.sin(beta_rad) <= 1: return -1 # Physical limit M1n > 1
        num = 2 * (1/math.tan(beta_rad)) * (m_val**2 * math.sin(beta_rad)**2 - 1)
        den = m_val**2 * (g + math.cos(2 * beta_rad)) + 2
        return math.atan(num / den)

    # Find wave angle (beta) and turn angle (theta)
    beta, theta = 0, 0
    if input_type == "theta":
        theta_deg = input_value
        theta = math.radians(theta_deg)
        if is_weak is None:
            raise ValueError("Must specify weak/strong solution for turn angle input.")

        beta_min = math.asin(1 / m1)
        
        # Find theta_max numerically
        gr = (math.sqrt(5) + 1) / 2
        a, b = beta_min, math.pi / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        while abs(b-a) > 1e-9:
            if theta_from_beta(c,m1,gamma) > theta_from_beta(d,m1,gamma): b=d
            else: a=c
            c = b - (b - a) / gr
            d = a + (b - a) / gr
        beta_at_theta_max = (a+b)/2
        theta_max = theta_from_beta(beta_at_theta_max, m1, gamma)

        if theta > theta_max:
            raise ValueError(f"Turn angle > max possible of {math.degrees(theta_max):.2f}°")

        # Bisection to find beta for the given theta
        def root_func(b_rad): return theta_from_beta(b_rad, m1, gamma) - theta
        low = beta_min + 1e-9
        high = beta_at_theta_max
        if not is_weak: # Strong shock
            low = beta_at_theta_max
            high = math.pi/2 - 1e-9

        while high - low > 1e-9:
            mid = (low+high)/2
            if root_func(low) * root_func(mid) < 0: high = mid
            else: low = mid
        beta = (low+high)/2

    elif input_type == "beta":
        beta_deg = input_value
        beta = math.radians(beta_deg)
        beta_min = math.asin(1 / m1)
        if beta <= beta_min:
            raise ValueError(f"Wave angle must be > Mach angle ({math.degrees(beta_min):.2f}°)")
        theta = theta_from_beta(beta, m1, gamma)

    elif input_type == "m1n":
        m1n_val = input_value
        if not 1 < m1n_val <= m1:
            raise ValueError(f"M1n must be between 1 and M1 ({m1:.4f}).")
        beta = math.asin(m1n_val / m1)
        theta = theta_from_beta(beta, m1, gamma)

    else:
        raise ValueError("Invalid input type for Oblique Shock. Use 'theta', 'beta', or 'm1n'.")

    # Compute all properties from M1, beta, and theta
    m1n = m1 * math.sin(beta)
    m2n = math.sqrt((2 + (gamma - 1) * m1n**2) / (2 * gamma * m1n**2 - (gamma - 1)))
    m2 = m2n / math.sin(beta - theta)
    p2_p1 = 1 + 2 * gamma / (gamma + 1) * (m1n**2 - 1)
    rho2_rho1 = (gamma + 1) * m1n**2 / ((gamma - 1) * m1n**2 + 2)
    t2_t1 = p2_p1 / rho2_rho1
    po2_po1 = (((gamma + 1) * m1n**2) / ((gamma - 1) * m1n**2 + 2)) ** (gamma / (gamma - 1)) * \
              ((gamma + 1) / (2 * gamma * m1n**2 - (gamma - 1))) ** (1 / (gamma - 1))

    return {
        "M1": m1, "M2": m2, "Wave ang": math.degrees(beta),
        "Turn ang": math.degrees(theta), "p2/p1": p2_p1, "rho2/rho1": rho2_rho1,
        "T2/T1": t2_t1, "Po2/Po1": po2_po1, "M1n": m1n, "M2n": m2n,
    }

# -----------------------------
# Flask API Route
# -----------------------------
@app.route("/calculate", methods=["POST"])
def calculate():
    """Main API endpoint to handle all calculation requests."""
    data = request.get_json()
    g = float(data.get("gamma", 1.4))
    flow_type = data.get("flow_type")
    results = {}

    try:
        if flow_type == "Isentropic":
            it = data.get("isen_type")
            iv = float(data.get("isen_value"))
            is_sub = data.get("isen_sub", None)
            results = isentropic_calculate(g, it, iv, is_sub)
        
        elif flow_type == "Normal Shock":
            it = data.get("ns_type")
            iv = float(data.get("ns_value"))
            results = normal_shock_calculate(g, it, iv)

        elif flow_type == "Oblique Shock":
            m1 = float(data.get("os_m1"))
            it = data.get("os_type")
            iv = float(data.get("os_value"))
            is_weak = data.get("os_weak", None)
            results = oblique_shock_calculate(g, m1, it, iv, is_weak)
        
        else:
            return jsonify({"error": "Invalid flow type specified"}), 400

        # Round all numerical results for clean output
        final_results = {}
        for key, val in results.items():
            if val is None:
                final_results[key] = "N/A"
            elif isinstance(val, (int, float)):
                final_results[key] = round(val, 9)
            else:
                final_results[key] = val
        return jsonify(final_results)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected calculation error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)


