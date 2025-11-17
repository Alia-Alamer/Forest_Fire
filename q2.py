import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Daten einlesen
forestFire_r1 = pd.read_csv("forestfire_region1.csv")
forestFire_r2 = pd.read_csv("forestfire_region2.csv")

forestFire_r1["region"] = 1
forestFire_r2["region"] = 2

forestfire = pd.concat([forestFire_r1, forestFire_r2], ignore_index=True)

# Spaltennamen säubern
forestfire.columns = [x.strip() for x in forestfire.columns]

# Fehlende Werte entfernen
forestfire = forestfire.dropna()

# Datentypen korrigieren
forestfire["DC"] = forestfire["DC"].astype(float)
forestfire["FWI"] = forestfire["FWI"].astype(float)


print("FWI MODELLIERUNG")
print("=" * 60)

# MODELL 1: Einfaches Modell
print("Modell 1: Einfaches Modell")
model1 = smf.ols('FWI ~ FFMC + DMC + DC + ISI + BUI', data=forestfire).fit()
print("R²:", model1.rsquared)

# MODELL 2: Mit quadratischen Termen
print("\nModell 2: Mit quadratischen Termen (Krümmungen)")
model2 = smf.ols('FWI ~ FFMC + I(FFMC**2) + DMC + DC + ISI + I(ISI**2) + BUI + I(BUI**2)', 
                 data=forestfire).fit()
print("R²:", model2.rsquared)

verbesserung = (model2.rsquared - model1.rsquared) * 100
print("\nVerbesserung:", round(verbesserung, 1), "Prozentpunkte")

# Residual Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(model1.fittedvalues, model1.resid, alpha=0.5)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_xlabel('Vorhergesagte Werte')
axes[0].set_ylabel('Fehler')
axes[0].set_title('Modell 1')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(model2.fittedvalues, model2.resid, alpha=0.5, color='green')
axes[1].axhline(y=0, color='red', linestyle='--')
axes[1].set_xlabel('Vorhergesagte Werte')
axes[1].set_ylabel('Fehler')
axes[1].set_title('Modell 2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Vorhergesagt vs Tatsächlich
plt.figure(figsize=(8, 6))
plt.scatter(forestfire['FWI'], model2.fittedvalues, alpha=0.6)
plt.plot([0, 35], [0, 35], 'r--', linewidth=2)
plt.xlabel('Tatsächliche FWI')
plt.ylabel('Vorhergesagte FWI')
plt.title('Vorhergesagt vs Tatsächlich')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG")
print("=" * 60)
print("\nModell 1 R²:", round(model1.rsquared, 3))
print("Modell 2 R²:", round(model2.rsquared, 3))
print("\nModell 2 ist besser weil es die Krümmungen berücksichtigt")
print("=" * 60)