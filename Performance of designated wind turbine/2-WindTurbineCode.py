import os
import numpy as np
import pandas as pd
from scipy.special import gamma

# ============================================================
# USER SETTINGS
# ============================================================
input_file = "PMDWindSpeedAt10m.xlsx"
sheet_name = "Sheet01"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

rho = 1.225
z_ref = 10.0
co2_factor = 0.606  # tCO2/MWh

# Economic assumptions (project-level; same for all turbines)
life_years = 20
discount_rate = 0.08
inflation_rate = 0.06
o_and_m_fraction = 0.025
other_initial_fraction = 0.30
salvage_fraction = 0.10

# Sensitivity for specific cost (USD/kW)
# low, medium, high cases
# You can edit these values if your supervisor wants different scenarios
cost_scenarios = {
    "low":    {"small": 2200, "medium": 1500, "large": 1000},
    "medium": {"small": 2550, "medium": 1900, "large": 1300},
    "high":   {"small": 2900, "medium": 2300, "large": 1600},
}

# Turbines from your Table S6
turbines = {
    30: {
        "model": "Aircon 10/10 kW",
        "Pr_MW": 0.010,
        "Vr": 11.0,
        "Vc": 2.5,
        "Vf": 32.0,
        "D": 7.1,
        "A": 39.6,
        "cost_class": "small"
    },
    50: {
        "model": "NM48/750",
        "Pr_MW": 0.750,
        "Vr": 16.0,
        "Vc": 3.5,
        "Vf": 25.0,
        "D": 48.2,
        "A": 1825.0,
        "cost_class": "large"
    },
    80: {
        "model": "V80/1800",
        "Pr_MW": 1.800,
        "Vr": 15.0,
        "Vc": 3.5,
        "Vf": 30.0,
        "D": 80.0,
        "A": 5027.0,
        "cost_class": "large"
    },
    100: {
        "model": "MM100",
        "Pr_MW": 2.000,
        "Vr": 11.5,
        "Vc": 3.5,
        "Vf": 22.0,
        "D": 100.0,
        "A": 7854.0,
        "cost_class": "large"
    }
}

# ============================================================
# FUNCTIONS
# ============================================================
def compute_a_from_v10(v10, zref=10.0):
    v10 = np.asarray(v10, dtype=float)
    v10 = np.where(v10 > 0, v10, np.nan)
    denom = 1.0 - 0.0088 * np.log(zref / 10.0)
    return (0.37 - 0.088 * np.log(v10)) / denom

def power_curve(V, Vc, Vr, Vf, Pr_MW):
    V = np.asarray(V, dtype=float)
    P = np.zeros_like(V, dtype=float)
    m1 = (V >= Vc) & (V < Vr)
    m2 = (V >= Vr) & (V < Vf)
    P[m1] = Pr_MW * (V[m1]**2 - Vc**2) / (Vr**2 - Vc**2)
    P[m2] = Pr_MW
    return np.clip(P, 0.0, Pr_MW)

def weibull_fit_moments(V):
    V = np.asarray(V, dtype=float)
    V = V[np.isfinite(V) & (V > 0)]
    if len(V) < 2:
        return np.nan, np.nan
    mean_v = np.mean(V)
    std_v = np.std(V, ddof=1)
    if mean_v <= 0 or std_v <= 0:
        return np.nan, np.nan
    k = (std_v / mean_v) ** -1.086
    c = mean_v / gamma(1.0 + 1.0 / k)
    return k, c

def zmax_pallabazzer(Pr_MW, rho, A, Vc, Vr):
    Pr_W = Pr_MW * 1e6
    return 0.77 * Pr_W / (rho * A * Vc * (Vr**2 - Vc**2))

def pv_cost(I, Com1, i, r, n, S):
    return I + Com1 * ((1 + i) / (r - i)) * (1 - ((1 + i) / (1 + r))**n) - S * ((1 + i) / (1 + r))**n

def lcoe_from_pvc(PVC, Pr_MW, CF):
    return PVC / (8760.0 * Pr_MW * CF)

def specific_cost_by_class(cost_class, scenario):
    return cost_scenarios[scenario][cost_class]

# ============================================================
# READ DATA
# ============================================================
df = pd.read_excel(input_file, sheet_name=sheet_name)
df = df.iloc[:, :3].copy()
df.columns = ["Year", "col2", "V10"]
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["V10"] = pd.to_numeric(df["V10"], errors="coerce")
df = df.dropna(subset=["Year", "V10"]).copy()
df["Year"] = df["Year"].astype(int)

# Calculate shear exponent from Eq. 4
df["a"] = compute_a_from_v10(df["V10"].to_numpy(), zref=z_ref)

# ============================================================
# MAIN CALCULATIONS
# ============================================================
annual_rows = df[["Year", "V10", "a"]].copy()
all_results = []

for hub, turb in turbines.items():
    Vh = df["V10"].to_numpy() * (hub / z_ref) ** df["a"].to_numpy()
    P = power_curve(Vh, turb["Vc"], turb["Vr"], turb["Vf"], turb["Pr_MW"])

    annual_energy_MWh = P.mean() * 8760.0
    cf = P.mean() / turb["Pr_MW"]
    mean_v = Vh.mean()
    std_v = Vh.std(ddof=1)
    k_weibull, c_weibull = weibull_fit_moments(Vh)

    wpd = 0.5 * rho * np.mean(Vh**3)
    wed = wpd * 8760.0 / 1e6
    available_energy_MWh = wed * turb["A"]

    ZT = annual_energy_MWh / available_energy_MWh if available_energy_MWh > 0 else np.nan
    Zmax = zmax_pallabazzer(turb["Pr_MW"], rho, turb["A"], turb["Vc"], turb["Vr"])
    e = annual_energy_MWh / (Zmax * available_energy_MWh) if (available_energy_MWh > 0 and Zmax > 0) else np.nan

    co2_reduction_t = annual_energy_MWh * co2_factor
    specific_aep = annual_energy_MWh / turb["Pr_MW"]  # MWh/MW = full-load hours

    annual_rows[f"V_{hub}m"] = Vh
    annual_rows[f"P_{hub}m_MW"] = P

    for scenario in ["low", "medium", "high"]:
        spec_cost = specific_cost_by_class(turb["cost_class"], scenario)
        turbine_cost_usd = turb["Pr_MW"] * 1000.0 * spec_cost
        initial_investment_usd = turbine_cost_usd * (1 + other_initial_fraction)
        om_first_year_usd = turbine_cost_usd * o_and_m_fraction
        salvage_value_usd = turbine_cost_usd * salvage_fraction
        PVC = pv_cost(initial_investment_usd, om_first_year_usd, inflation_rate, discount_rate, life_years, salvage_value_usd)
        LCOE = lcoe_from_pvc(PVC, turb["Pr_MW"], cf)

        all_results.append({
            "HubHeight_m": hub,
            "TurbineModel": turb["model"],
            "CostScenario": scenario,
            "RatedPower_MW": turb["Pr_MW"],
            "CostClass": turb["cost_class"],
            "SpecificCost_USD_kW": spec_cost,
            "MeanWindSpeed_mps": mean_v,
            "WindSpeedStd_mps": std_v,
            "Weibull_k": k_weibull,
            "Weibull_c_mps": c_weibull,
            "AnnualEnergy_MWh": annual_energy_MWh,
            "SpecificAEP_MWh_per_MW": specific_aep,
            "CapacityFactor_pct": cf * 100.0,
            "WindPowerDensity_W_m2": wpd,
            "WindEnergyDensity_MWh_m2_yr": wed,
            "Zmax": Zmax,
            "EnergyEfficiency_ZT": ZT,
            "SiteEffectiveness_e": e,
            "CO2_Reduction_tCO2": co2_reduction_t,
            "TurbineCost_USD": turbine_cost_usd,
            "InitialInvestment_USD": initial_investment_usd,
            "OandM_FirstYear_USD": om_first_year_usd,
            "SalvageValue_USD": salvage_value_usd,
            "PVC_USD": PVC,
            "LCOE_USD_MWh": LCOE
        })

results = pd.DataFrame(all_results)

# Rankings within each scenario
results["EnergyRank"] = results.groupby("CostScenario")["AnnualEnergy_MWh"].rank(ascending=False, method="min")
results["CF_Rank"] = results.groupby("CostScenario")["CapacityFactor_pct"].rank(ascending=False, method="min")
results["LCOE_Rank"] = results.groupby("CostScenario")["LCOE_USD_MWh"].rank(ascending=True, method="min")

# ============================================================
# OUTPUT TABLES
# ============================================================
results.to_csv(os.path.join(output_dir, "economic_sensitivity_results.csv"), index=False)
annual_rows.to_csv(os.path.join(output_dir, "annual_detail_with_power.csv"), index=False)

with pd.ExcelWriter(os.path.join(output_dir, "economic_sensitivity_results.xlsx"), engine="openpyxl") as writer:
    results.to_excel(writer, index=False, sheet_name="results")
    annual_rows.to_excel(writer, index=False, sheet_name="annual_detail")

main_table = results[[
    "CostScenario", "TurbineModel", "HubHeight_m", "RatedPower_MW",
    "MeanWindSpeed_mps", "AnnualEnergy_MWh", "SpecificAEP_MWh_per_MW",
    "CapacityFactor_pct", "CO2_Reduction_tCO2", "LCOE_USD_MWh"
]].copy()

main_table.columns = [
    "Scenario", "Turbine model", "Hub height (m)", "Rated power (MW)",
    "Mean wind speed (m/s)", "AEP (MWh/yr)", "Specific AEP (MWh/MW)",
    "Capacity factor (%)", "CO2 reduction (t/yr)", "LCOE (USD/MWh)"
]

technical_table = results[[
    "CostScenario", "TurbineModel", "HubHeight_m", "SpecificCost_USD_kW",
    "WindPowerDensity_W_m2", "WindEnergyDensity_MWh_m2_yr", "Weibull_k",
    "Weibull_c_mps", "Zmax", "EnergyEfficiency_ZT", "SiteEffectiveness_e",
    "TurbineCost_USD", "PVC_USD"
]].copy()

technical_table.columns = [
    "Scenario", "Turbine model", "Hub height (m)", "Specific cost (USD/kW)",
    "WPD (W/m2)", "WED (MWh/m2/yr)", "Weibull k", "Weibull c (m/s)",
    "Zmax", "ZT", "e", "Turbine cost (USD)", "PVC (USD)"
]

main_table.to_csv(os.path.join(output_dir, "paper_table_main.csv"), index=False)
technical_table.to_csv(os.path.join(output_dir, "paper_table_technical.csv"), index=False)

print(main_table.to_string(index=False))
print("\nTECHNICAL TABLE\n")
print(technical_table.to_string(index=False))
