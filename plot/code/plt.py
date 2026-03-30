import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Provided data
data = {
    "Function": ["Sphere","Manhattan","Quartic","Sine","Composite Sine","Rastrigin",
                 "Sine with Deviation", "Exponential", "Sum Cosine with Shift","Sum Square","Dixon Price","Zakharov"],
    
    "QEA_Best": [8.49E+00, 5.24E+00, 1.78E+02, 7.06E+02, 6.72E+01, 4.47E+01,
                 7.49E+00, 9.26E+02, 1.21E+03, 4.03E+01, 5.86E+02, 3.66E+01],
    
    "QPSO_Best": [0.00E+00, 0.00E+00, 7.83E+02, 8.85E-02, 8.42E+01, 1.50E+00,
                  1.00E+00, 1.00E+03, 1.18E+03, 0.00E+00, 7.73E-08, 1.37E-112],
    
    "QHTS_Best": [9.30E-181, 2.92E-90, 5.28E+02, 7.52E-181, 4.23E+01, 0.00E+00,
                  1.00E+00, 8.11E+01, 1.17E+02, 8.23E-180, 9.09E+00, 3.74E-180],
    
    "QEA_Time": [6.90E+01, 8.55E+01, 8.81E+01, 8.63E+01, 7.29E+01, 7.25E+01,
                 7.54E+01, 7.21E+01, 7.49E+01, 7.47E+01, 8.00E+01, 9.43E+01],
    
    "QPSO_Time": [2.16E+01, 2.48E+01, 3.05E+01, 2.62E+01, 5.06E+01, 3.44E+01,
                  5.68E+01, 2.80E+01, 2.94E+01, 2.78E+01, 3.29E+01, 2.71E+01],
    
    "QHTS_Time": [3.22E+00, 3.01E+00, 4.60E+00, 1.37E+00, 1.55E+00, 1.61E+00,
                  1.60E+00, 1.51E+00, 1.86E+00, 1.97E+00, 2.37E+00, 2.72E+00]
}



df = pd.DataFrame(data)

# Plot 1: Best Values Comparison with thicker lines
# Ensure best-value columns are positive for log scale (replace <=0 with tiny positive)
for col in ("QEA_Best", "QPSO_Best", "QHTS_Best"):
    arr = np.array(df[col], dtype=float)
    if np.any(arr <= 0):
        arr = np.where(arr <= 0, np.finfo(float).tiny, arr)
        df[col] = arr

plt.figure(figsize=(12, 6))
plt.plot(df["Function"], df["QEA_Best"], marker='p', label="QEA Best", linewidth=2)
plt.plot(df["Function"], df["QPSO_Best"], marker='p', label="QPSO Best", linewidth=2)
plt.plot(df["Function"], df["QHTS_Best"], marker='p', label="QHTS Best", linewidth=2)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Benchmark Function")
plt.ylabel("Best Fitness Value")
plt.title("Comparison of Best Fitness Values (D = 10)")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()

# Plot 2: Time Comparison with thicker lines
plt.figure(figsize=(12, 6))
plt.plot(df["Function"], df["QEA_Time"], marker='s', label="QEA Time", linewidth=2)
plt.plot(df["Function"], df["QPSO_Time"], marker='s', label="QPSO Time", linewidth=2)
plt.plot(df["Function"], df["QHTS_Time"], marker='s', label="QHTS Time", linewidth=2)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Benchmark Function")
plt.ylabel("Execution Time (seconds)")
plt.title("Comparison of Execution Time (D = 10)")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()
