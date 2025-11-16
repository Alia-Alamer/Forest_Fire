import pandas as pd
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

forestFire_r1 = pd.read_csv("forestfire_region1.csv")
forestFire_r2 = pd.read_csv("forestfire_region2.csv")


forestFire_r1["region"] = 1
forestFire_r2["region"] = 2

forestfire = pd.concat([forestFire_r1, forestFire_r2], ignore_index=True).reset_index(drop=True)

#print(forestfire.head())

print(forestfire.columns)
forestfire.columns = [x.strip() for x in forestfire.columns]
print(forestfire.columns)

forestfire["datetime"] = pd.to_datetime(forestfire[['year', 'month', 'day']], format="%Y-%m-%d")

forestfire = forestfire.dropna()
forestfire["Classes"].apply(lambda x: x.strip()).astype("category")
forestfire["Classes"] = (forestfire["Classes"].apply(lambda x: x.strip()) == "fire").astype(int)

forestfire["DC"] = forestfire["DC"].astype(float)
forestfire["FWI"] = forestfire["FWI"].astype(float)


plt.figure(figsize=(10, 6))
plt.scatter(forestfire["ISI"], forestfire["FWI"], alpha=0.7, color='red', s=50)
plt.xlabel('ISI', fontsize=12)
plt.ylabel('FWI', fontsize=12)
plt.title('Zusammenhang zwischen ISI UND FWI', fontsize=14)
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(forestfire["BUI"], forestfire["FWI"], alpha=0.7, color='red', s=50)
plt.xlabel('BUI', fontsize=12)
plt.ylabel('FWI', fontsize=12)
plt.title('Zusammenhang zwischen BUI UND FWI', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(forestfire["FFMC"], forestfire["FWI"], alpha=0.7, color='red', s=50)
plt.xlabel('FFMC', fontsize=12)
plt.ylabel('FWI', fontsize=12)
plt.title('Zusammenhang zwischen FFMC UND FWI', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(forestfire["DMC"], forestfire["FWI"], alpha=0.7, color='red', s=50)
plt.xlabel('DMC', fontsize=12)
plt.ylabel('FWI', fontsize=12)
plt.title('Zusammenhang zwischen DMC UND FWI', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(forestfire["DC"], forestfire["FWI"], alpha=0.7, color='red', s=50)
plt.xlabel('DC', fontsize=12)
plt.ylabel('FWI', fontsize=12)
plt.title('Zusammenhang zwischen DC UND FWI', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

sns.pairplot(forestfire[["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]], y_vars='FWI')
plt.show()


print("\nCorrelation with FWI (Fire Indices Only):")
for col in ["FFMC", "DMC", "DC", "ISI", "BUI"]:
    corr = forestfire[col].corr(forestfire["FWI"])
    print(f"{col}: {corr:.3f}")


















#plt.figure(figsize=(10, 6))
#plt.scatter(forestfire["Temperature"], forestfire["FWI"], alpha=0.7, color='red', s=50)
#plt.xlabel('Temperature', fontsize=12)
#plt.ylabel('FWI', fontsize=12)
#plt.title('Zusammenhang zwischen Temperature UND FWI', fontsize=14)
#plt.grid(True, alpha=0.3)

#plt.tight_layout()
#plt.show()


#plt.figure(figsize=(10, 6))
#plt.scatter(forestfire["RH"], forestfire["FWI"], alpha=0.7, color='red', s=50)
#plt.xlabel('RH', fontsize=12)
#plt.ylabel('FWI', fontsize=12)
#plt.title('Zusammenhang zwischen RH UND FWI', fontsize=14)
#plt.grid(True, alpha=0.3)

#plt.tight_layout()
#plt.show()


#plt.figure(figsize=(10, 6))
#plt.scatter(forestfire["Ws"], forestfire["FWI"], alpha=0.7, color='red', s=50)
#plt.xlabel('Ws', fontsize=12)
#plt.ylabel('FWI', fontsize=12)
#plt.title('Zusammenhang zwischen Ws UND FWI', fontsize=14)
#plt.grid(True, alpha=0.3)
#
#plt.tight_layout()
#plt.show()


#plt.figure(figsize=(10, 6))
#plt.scatter(forestfire["Rain"], forestfire["FWI"], alpha=0.7, color='red', s=50)
#plt.xlabel('Rain', fontsize=12)
#plt.ylabel('FWI', fontsize=12)
#plt.title('Zusammenhang zwischen Rain UND FWI', fontsize=14)
#plt.grid(True, alpha=0.3)

#plt.tight_layout()
#plt.show()