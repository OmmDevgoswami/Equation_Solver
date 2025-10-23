import sympy as sp
import numpy as np
from scipy.optimize import least_squares
import streamlit as st
import plotly.graph_objects as go

# === 1. DEFINE SYMBOLS ===
x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, r = sp.symbols('x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 r')
alpha, B, eps, z = sp.symbols('alpha B eps z', real=True)

# === 2. DEFINE EQUATIONS ===
eqs = [
    x4**2 + y4**2 - z**2,
    (x5 - x4)**2 + (y5 - y4)**2 - (x1**2 + y1**2),
    (x3 - x2)**2 + (y3 - y2)**2 - r**2,
    (x5 - x2)**2 + (y5 - y2)**2 - r**2,
    (x6 - x2)**2 + (y6 - y2)**2 - r**2,
    (x2 - x1)**2 + (y2 - y1)**2 + (x5 - x2)**2 + (y5 - y2)**2 - z**2,
    (x6 - x1)**2 + (y6 - y1)**2 - (
        (x6 - x2)**2 + (y6 - y2)**2 + (x2 - x1)**2 + (y2 - y1)**2
        + 2 * sp.sqrt((x6 - x2)**2 + (y6 - y2)**2) *
            sp.sqrt((x2 - x1)**2 + (y2 - y1)**2) * sp.cos(B)
    ),
    (x6 - x5)**2 + (y6 - y5)**2 - (
        (x6 - x2)**2 + (y6 - y2)**2 + (x5 - x2)**2 + (y5 - y2)**2
        - 2 * sp.sqrt((x6 - x2)**2 + (y6 - y2)**2) *
            sp.sqrt((x5 - x2)**2 + (y5 - y2)**2) * sp.cos(B)
    ),
    (x5 - x3)**2 + (y5 - y3)**2 - (
        (x5 - x2)**2 + (y5 - y2)**2 + (x3 - x2)**2 + (y3 - y2)**2
        - 2 * sp.sqrt((x5 - x2)**2 + (y5 - y2)**2) *
            sp.sqrt((x3 - x2)**2 + (y3 - y2)**2) * sp.cos(eps)
    ),
    (x6 - x3)**2 + (y6 - y3)**2 - (
        (x6 - x2)**2 + (y6 - y2)**2 + (x3 - x2)**2 + (y3 - y2)**2
        - 2 * sp.sqrt((x6 - x2)**2 + (y6 - y2)**2) *
            sp.sqrt((x3 - x2)**2 + (y3 - y2)**2) * sp.cos(B + eps)
    ),
    (x3 - x1)**2 + (y3 - y1)**2 - (
        (x3 - x2)**2 + (y3 - y2)**2 + (x2 - x1)**2 + (y2 - y1)**2
        + 2 * sp.sqrt((x3 - x2)**2 + (y3 - y2)**2) *
            sp.sqrt((x2 - x1)**2 + (y2 - y1)**2) * sp.cos(eps)
    ),
    (x5 - x3)**2 + (y5 - y3)**2 - (
        (x4 - x3)**2 + (y4 - y3)**2 + (x5 - x4)**2 + (y5 - y4)**2
        - 2 * sp.sqrt((x4 - x3)**2 + (y4 - y3)**2) *
            sp.sqrt((x5 - x4)**2 + (y5 - y4)**2) * sp.cos(alpha)
    ),
    x2**2 + y2**2 - ((x2 - x1)**2 + (y2 - y1)**2 + x1**2 + y1**2),
]

variables = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, r]
f_numeric = sp.lambdify((variables, alpha, B, eps, z), eqs, "numpy")

# === 3. HYBRID SOLVER FUNCTION ===
def hybrid_solver(alpha_val, B_val, eps_val, z_val, degree=True):
    if degree:
        alpha_val = np.radians(alpha_val)
        B_val = np.radians(B_val)
        eps_val = np.radians(eps_val)
    
    def residual(vars):
        return np.array(f_numeric(vars, alpha_val, B_val, eps_val, z_val), dtype=float)
    
    np.random.seed(42)
    init_guess = np.random.uniform(-1, 1, len(variables))
    
    res = least_squares(residual, init_guess, max_nfev=30000, xtol=1e-12, ftol=1e-12)
    
    solution = {str(var): round(val, 5) for var, val in zip(variables, res.x)}
    return solution, res.success

# === 4. STREAMLIT UI ===
st.set_page_config("GeoSolver V4", layout="wide")
st.title("📐 GeoSolver V4 – Multi-Angle Coordinate Calculator")
st.sidebar.header("Input Parameters")

alpha_input = st.sidebar.number_input("α (Alpha)", value=30.0)
B_input     = st.sidebar.number_input("β (Beta)", value=45.0)
eps_input   = st.sidebar.number_input("ε (Epsilon)", value=60.0)
z_input     = st.sidebar.number_input("Z", value=1.0)
degree_input = st.sidebar.radio("Angle Unit", ["Degrees", "Radians"]) == "Degrees"

if st.sidebar.button("Solve"):
    solution, success = hybrid_solver(alpha_input, B_input, eps_input, z_input, degree=degree_input)
    
    if success:
        st.success("✅ Solution Found!")
        st.subheader("Variables")
        st.table(solution)
        
        # === 5. PLOTLY GRAPH ===
        fig = go.Figure()
        # extract coordinates
        points = ['x1','y1','x2','y2','x3','y3','x4','y4','x5','y5','x6','y6']
        coords = [(solution[points[i]], solution[points[i+1]]) for i in range(0,len(points),2)]
        x_vals = [c[0] for c in coords]
        y_vals = [c[1] for c in coords]
        labels = ['P1','P2','P3','P4','P5','P6']
        
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers+text', text=labels, textposition='top center', marker=dict(size=12,color='red')))
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='blue', width=2)))
        
        fig.update_layout(title="📍 Geometry Plot", xaxis_title="X", yaxis_title="Y", width=800, height=600)
        st.plotly_chart(fig)
    else:
        st.error("❌ Solver failed to converge. Try changing inputs or rerun.")
