import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA

# === Load the data ===
file_path = 'anonymized_data.xlsx'  
df = pd.read_excel(file_path)
df = df.rename(columns={'Type of test': 'IV'})

# === Custom Formatter ===
def fmt(val, places=2):
    """Force formatting to exact decimal places."""
    return f"{val:.{places}f}"

# === Multivariate Test ===
manova = MANOVA.from_formula('Anxiety + Stress + Spirituality ~ IV', data=df)
summary = str(manova.mv_test())

# Extract only IV-related multivariate lines
lines = summary.splitlines()
iv_start = next(i for i, line in enumerate(lines) if 'IV' in line and 'Value' in line)
iv_section = lines[iv_start + 1 : iv_start + 5]

# === Print Output Matching JAMOVI ===
print("\n" + "="*70)
print("MANCOVA\n")

# --- Multivariate Tests ---
print("Multivariate Tests")
print("{:<25} {:>7} {:>6} {:>4} {:>4} {:>6}".format("Type of Test", "value", "F", "df1", "df2", "p"))

for line in iv_section:
    parts = [p for p in line.strip().split() if p]
    if len(parts) >= 6:
        test_name = " ".join(parts[:-5])
        value, fval, df1, df2, pval = parts[-5:]
        print("{:<25} {:>7} {:>6} {:>4} {:>4} {:>6}".format(
            test_name,
            fmt(float(value), 4),   
            fmt(float(fval), 2),    
            int(float(df1)),        
            int(float(df2)),        
            fmt(float(pval), 3)     
        ))

# --- Univariate Tests ---
print("\nUnivariate Tests")
print("{:<18} {:>15} {:>4} {:>13} {:>6} {:>6}".format(
    "Dependent Variable", "Sum of Squares", "df", "Mean Square", "F", "p"))

for dv in ['Anxiety', 'Spirituality', 'Stress']:
    model = ols(f'{dv} ~ C(IV)', data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)

    ss_effect = anova.loc['C(IV)', 'sum_sq']
    df_effect = anova.loc['C(IV)', 'df']
    ms_effect = ss_effect / df_effect
    f_val = anova.loc['C(IV)', 'F']
    p_val = anova.loc['C(IV)', 'PR(>F)']

    print("{:<18} {:>15} {:>4} {:>13} {:>6} {:>6}".format(
        dv,
        fmt(ss_effect, 0),              
        int(df_effect),                 
        fmt(ms_effect, 1),              
        fmt(f_val, 2),                  
        fmt(p_val, 3)                   
    ))

# --- Residuals Block ---
print("\nResiduals")
print("{:<18} {:>15} {:>4} {:>13}".format("Dependent Variable", "Sum of Squares", "df", "Mean Square"))

for dv in ['Anxiety', 'Spirituality', 'Stress']:
    model = ols(f'{dv} ~ C(IV)', data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)

    ss_resid = anova.loc['Residual', 'sum_sq']
    df_resid = anova.loc['Residual', 'df']
    ms_resid = ss_resid / df_resid

    print("{:<18} {:>15} {:>4} {:>13}".format(
        dv,
        fmt(ss_resid, 0),              
        int(df_resid),                 
        fmt(ms_resid, 1)               
    ))

print("="*70 + "\n")
