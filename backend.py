# backend.py (Corrected)
import math
# CORRECTED: Combined imports into one line
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# CORRECTED: Use a single, standard route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Helper functions
# (Your calculation functions remain the same)
# -----------------------------
def isentropic_calculate(gamma, input_type, input_value, is_sub=None):
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

    def t_t0_star(g):
        return 2 / (g + 1)

    def p_p0_star(g):
        return (2 / (g + 1)) ** (g / (g - 1))

    def rho_rho0_star(g):
        return (2 / (g + 1)) ** (1 / (g - 1))

    def bisection_solve(func, target, low, high, tol=1e-8):
        while high - low > tol:
            mid = (low + high) / 2
            val = func(mid)
            if val < target:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    # Find M
    if input_type == "mach":
        m = input_value
    elif input_type == "mach_angle":
        m = 1 / math.sin(math.radians(input_value))
    elif input_type == "pm_angle":
        def pm_func(m_val):
            return pm_angle(m_val, gamma)
        m = bisection_solve(pm_func, input_value, 1.0001, 50)
    elif input_type == "p_p0":
        t_t0 = input_value ** ((gamma - 1) / gamma)
        m = math.sqrt(2 / (gamma - 1) * (1 / t_t0 - 1))
    elif input_type == "rho_rho0":
        t_t0 = input_value ** (gamma - 1)
        m = math.sqrt(2 / (gamma - 1) * (1 / t_t0 - 1))
    elif input_type == "t_t0":
        t_t0 = input_value
        m = math.sqrt(2 / (gamma - 1) * (1 / t_t0 - 1))
    elif input_type == "a_a_star":
        if is_sub is None:
            raise ValueError("Specify subsonic (True) or supersonic (False)")
        def area_func(m_val):
            return area_ratio(m_val, gamma)
        if is_sub:
            m = bisection_solve(area_func, input_value, 0.0001, 0.9999)
        else:
            m = bisection_solve(area_func, input_value, 1.0001, 50)

    # Compute outputs
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
        "Mach": m,
        "Mach angle": mu,
        "P-M angle": nu,
        "p/p0": p_p0,
        "rho/rho0": rho_rho0,
        "T/T0": t_t0,
        "p/p*": p_p_star,
        "rho/rho*": rho_rho_star,
        "T/T*": t_t_star,
        "A/A*": a_a_star,
    }


def normal_shock_calculate(gamma, m1):
    m2 = math.sqrt((2 + (gamma - 1) * m1**2) / (2 * gamma * m1**2 - (gamma - 1)))
    p2_p1 = (2 * gamma * m1**2 - (gamma - 1)) / (gamma + 1)
    rho2_rho1 = ((gamma + 1) * m1**2) / ((gamma - 1) * m1**2 + 2)
    t2_t1 = p2_p1 / rho2_rho1
    po2_po1 = (((gamma + 1) * m1**2) / ((gamma - 1) * m1**2 + 2)) ** (gamma / (gamma - 1)) * \
              ((gamma + 1) / (2 * gamma * m1**2 - (gamma - 1))) ** (1 / (gamma - 1))
    return {
        "M1": m1,
        "M2": m2,
        "p2/p1": p2_p1,
        "rho2/rho1": rho2_rho1,
        "T2/T1": t2_t1,
        "Po2/Po1": po2_po1,
    }


def oblique_shock_calculate(gamma, m1, theta_deg, is_weak):
    theta = math.radians(theta_deg)

    def theta_from_beta(beta_rad, m_val, g):
        beta = beta_rad
        sinb2 = math.sin(beta)**2
        cot_beta = 1 / math.tan(beta)
        num = 2 * cot_beta * (m_val**2 * sinb2 - 1)
        den = m_val**2 * (g + math.cos(2 * beta)) + 2
        return math.atan(num / den)

    def golden_maximize(func, a, b, tol=1e-6):
        gr = (math.sqrt(5) + 1) / 2
        while abs(b - a) > tol:
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            if func(c) > func(d):
                b = d
            else:
                a = c
        return (a + b) / 2

    beta_min = math.asin(1 / m1)
    beta_max_pos = golden_maximize(lambda b: theta_from_beta(b, m1, gamma), beta_min, math.pi / 2 - 0.001)
    theta_max = theta_from_beta(beta_max_pos, m1, gamma)

    if theta > theta_max:
        raise ValueError("No solution: theta exceeds max possible.")

    def bisection_root(f, low, high, tol=1e-8):
        f_low = f(low)
        f_high = f(high)
        if f_low * f_high > 0:
            raise ValueError("No root in interval.")
        while high - low > tol:
            mid = (low + high) / 2
            f_mid = f(mid)
            if f_low * f_mid < 0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        return (low + high) / 2

    if is_weak:
        def f_weak(b):
            return theta_from_beta(b, m1, gamma) - theta
        beta = bisection_root(f_weak, beta_min + 0.0001, beta_max_pos - 0.0001)
    else:
        def f_strong(b):
            return theta_from_beta(b, m1, gamma) - theta
        beta = bisection_root(f_strong, beta_max_pos + 0.0001, math.pi / 2 - 0.0001)

    beta_deg = math.degrees(beta)
    m1n = m1 * math.sin(beta)
    m2n = math.sqrt((2 + (gamma - 1) * m1n**2) / (2 * gamma * m1n**2 - (gamma - 1)))
    m2 = m2n / math.sin(beta - theta)
    p2_p1 = 1 + 2 * gamma / (gamma + 1) * (m1n**2 - 1)
    rho2_rho1 = (gamma + 1) * m1n**2 / ((gamma - 1) * m1n**2 + 2)
    t2_t1 = p2_p1 / rho2_rho1
    po2_po1 = (((gamma + 1) * m1n**2) / ((gamma - 1) * m1n**2 + 2)) ** (gamma / (gamma - 1)) * \
              ((gamma + 1) / (2 * gamma * m1n**2 - (gamma - 1))) ** (1 / (gamma - 1))

    return {
        "M1": m1,
        "M2": m2,
        "Wave ang": beta_deg,
        "Turn ang": theta_deg,
        "p2/p1": p2_p1,
        "rho2/rho1": rho2_rho1,
        "T2/T1": t2_t1,
        "Po2/Po1": po2_po1,
        "M1n": m1n,
        "M2n": m2n,
    }


# -----------------------------
# Flask routes
# -----------------------------
@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()
    g = float(data.get("gamma", 1.4))
    flow_type = data.get("flow_type")

    try:
        if flow_type == "Isentropic":
            it = data.get("isen_type")
            iv = float(data.get("isen_value"))
            is_sub = data.get("isen_sub")
            results = isentropic_calculate(g, it, iv, is_sub)
        elif flow_type == "Normal Shock":
            m1 = float(data.get("ns_m1"))
            results = normal_shock_calculate(g, m1)
        elif flow_type == "Oblique Shock":
            m1 = float(data.get("os_m1"))
            theta = float(data.get("os_theta"))
            is_weak = data.get("os_weak")
            results = oblique_shock_calculate(g, m1, theta, is_weak)
        else:
            return jsonify({"error": "Invalid flow type"}), 400

        # Round results
        res = {}
        for k, v in results.items():
            if v is None:
                res[k] = "N/A"
            else:
                res[k] = round(v, 9)
        return jsonify(res)

    except ValueError as e:
        return jsonify({"error": str(e)})
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True)
