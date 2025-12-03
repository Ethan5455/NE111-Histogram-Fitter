import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import (norm, expon, gamma, beta, lognorm, weibull_min, uniform, chi2, logistic, rayleigh)

# Title
st.title("NE111 Histogram Fitter")

#Enter data
st.subheader("Enter data")

raw_text = st.text_area(
    "Paste data (numbers) separated by commas or spaces:",
    height=150,
    placeholder="e.g., 1.2, 3.4, 5.6 7.8 9.1"
)

st.write("Or upload a CSV file:")
file = st.file_uploader("Upload CSV", type=["csv"])

# ---- Parse text input ----
data_text = []

if raw_text.strip():
    text = raw_text.replace(",", " ").split()
    data_text = [0.0] * len(text)

    try:
        for i in range(len(text)):
            data_text[i] = float(text[i])

        st.success(f"Loaded {len(data_text)} values from the text box.")

    except ValueError:
        st.error("Some values were not valid numbers. Please check your input.")
        data_text = []

# ---- Parse CSV input ----
data_file = []

if file is not None:
    try:
        df = pd.read_csv(file)

        # Allow user to choose which column to use
        column = st.selectbox("Select column for data:", df.columns)

        # Convert chosen column to numeric list using pandas: gets rid of missing values and coverts the rest to floats
        data_file = df[column].dropna().astype(float).tolist()

        st.success(f"Loaded {len(data_file)} values from the CSV file.")

        # Optional: preview inside an expander
        with st.expander("Show CSV preview (first 5 rows)"):
            st.dataframe(df.head())

    except ValueError:
        st.error("Could not read data from the CSV file try again.")

# ---- Decide which data to use ----
if len(data_file) > 0:
    data = data_file
elif len(data_text) > 0:
    data = data_text
else:
    data = []

if len(data) == 0:
    st.info("Please enter data or upload a CSV file.")
    st.stop()

data_array = np.array(data, dtype=float)

# Data summary
st.subheader("Data summary")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Number of points", len(data_array))
with col2:
    st.metric("Mean", f"{data_array.mean():.3f}")
with col3:
    st.metric("Std dev", f"{data_array.std(ddof=1):.3f}")

st.caption(
    "Data source: " + ("CSV upload" if len(data_file) > 0 else "Text box")
)

with st.expander("Show raw data (optional)"):
    st.write(data_array)

#Choose a distribution
st.subheader("Choose a distribution")

distributions = {
    "Normal (norm)": norm,
    "Exponential (expon)": expon,
    "Gamma": gamma,
    "Beta": beta,
    "Lognormal (lognorm)": lognorm,
    "Weibull (weibull_min)": weibull_min,
    "Uniform": uniform,
    "Chi-Square (chi2)": chi2,
    "Logistic": logistic,
    "Rayleigh": rayleigh,
}

dist_name = st.selectbox("Distribution:", list(distributions.keys()))
dist_obj = distributions[dist_name]

st.write(f"You selected: **{dist_name}**")

#Automatic fit (with optional manual adjustment)
st.subheader("Fit distribution")

# Automatic fit to get starting parameters
params = dist_obj.fit(data_array)   # (shape(s)..., loc, scale)

# Split parameters into shape(s), loc, scale
num_shape = max(len(params) - 2, 0)
shape_params = list(params[:num_shape])
loc_param = float(params[-2])
scale_param = float(params[-1])

st.write("Fitted parameters (shape(s), loc, scale):")
st.write(params)

# Manual adjustment in sidebar
manual_mode = st.sidebar.checkbox("Enable manual parameter adjustment")

if manual_mode:
    st.sidebar.markdown("### Manual parameters")

    data_min = float(data_array.min())
    data_max = float(data_array.max())
    data_range = data_max - data_min if data_max > data_min else 1.0

    # ---- Shape parameter sliders (if any) ----
    for i in range(num_shape):
        base = float(shape_params[i])          # fitted shape value
        # allow about /5 to *5 around the fit, but never below 0.01
        shape_min = max(0.01, base / 5.0)
        shape_max = base * 5.0
        shape_params[i] = st.sidebar.slider(
            f"shape {i+1}",
            shape_min,
            shape_max,
            base
        )

    # ---- loc slider ----
    base_loc = float(loc_param)               # fitted loc
    # let loc move roughly one data_range around the fitted value
    loc_span = max(data_range, abs(base_loc) * 0.5)
    loc_min = base_loc - loc_span
    loc_max = base_loc + loc_span

    loc_param = st.sidebar.slider(
        "loc",
        loc_min,
        loc_max,
        base_loc
    )

    # ---- scale slider ----
    base_scale = float(scale_param)           # fitted scale
    # allow about /5 to *5 around the fitted value, keep > 0
    scale_min = max(0.01, base_scale / 5.0)
    scale_max = base_scale * 5.0

    scale_param = st.sidebar.slider(
        "scale",
        scale_min,
        scale_max,
        base_scale
    )

# Build the distribution actually used (auto-fit if manual off)
params_used = tuple(shape_params) + (loc_param, scale_param)
dist_used = dist_obj(*params_used)

# x-grid for plotting PDF
x_min = data_array.min()
x_max = data_array.max()
x = np.linspace(
    x_min - 0.1 * (x_max - x_min),
    x_max + 0.1 * (x_max - x_min),
    300
)

pdf = dist_used.pdf(x)

#Histogram with fitted curve
st.subheader("Histogram + curve")

fig, ax = plt.subplots()

ax.hist(
    data_array,
    bins="auto",
    density=True,
    alpha=0.6,
    edgecolor="black",
    label="Data histogram"
)

ax.plot(
    x,
    pdf,
    linewidth=2,
    label=f"{dist_name} PDF"
)

ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title(f"Histogram with {dist_name} fit")
ax.legend()


#Fit quality
# Histogram values (same binning as plot above)
hist_y, bin_edges = np.histogram(
    data_array,
    bins="auto",
    density=True
)

# Bin centers and PDF at those centers
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
pdf_at_centers = dist_used.pdf(bin_centers)

# Error metrics
errors = np.abs(hist_y - pdf_at_centers)
mean_err = errors.mean()
max_err = errors.max()

st.pyplot(fig)

st.subheader("Fit quality")
st.write(f"Mean absolute error: **{mean_err:.4f}**")
st.write(f"Maximum absolute error: **{max_err:.4f}**")
        