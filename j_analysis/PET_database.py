import numpy as np
import h5py
from pet_calc import calculate_PET_index_vec


class Pet:
    def __init__(self, mbody=70, age=35, height=175, activity=80.0, sex=1, clo=0.9):
        self.mbody = mbody
        self.age = age
        self.height = height
        self.activity = activity
        self.sex = sex
        self.clo = clo

    def to_dict(self):
        return {
            "mbody": self.mbody,
            "age": self.age,
            "height": self.height,
            "activity": self.activity,
            "sex": self.sex,
            "clo": self.clo,
        }

# Define standard people
body_types = {
    "standard_man": Pet(mbody=70, age=35, height=175, activity=80.0, sex=1, clo=0.9),
    "standard_woman": Pet(mbody=60, age=35, height=165, activity=80.0, sex=0, clo=0.9),
    "elderly_woman": Pet(mbody=55, age=75, height=160, activity=60.0, sex=0, clo=1.0),
    "young_child": Pet(mbody=20, age=5, height=110, activity=90.0, sex=0, clo=0.7),
}

# Grids with half-degree steps
wind_speeds = np.array([0.1, 2.0, 6.0])
rhs = np.arange(100, -1, -10)  # RH from 100 to 0
tmrts = np.arange(65.0, -0.1, -0.5)  # Tmrt from 65 to 0 in 0.5 steps
temps = np.arange(40.0, -0.1, -0.5)  # Air temperature from 40 to 0 in 0.5 steps

print(f"Wind Speeds: {wind_speeds.shape}, RHs: {rhs.shape}, Tmrts: {tmrts.shape}, Temps: {temps.shape}")

# Precompute and store PET values
with h5py.File("pet_lookup.h5", "w") as f:
    for body_name, pet in body_types.items():
        print(f"Generating PET values for: {body_name}")
        dset = f.create_dataset(
            name=body_name,
            shape=(len(wind_speeds), len(rhs), len(tmrts), len(temps)),
            dtype='f4',
        )
        for i, ws in enumerate(wind_speeds):
            for j, rh in enumerate(rhs):
                print(f"  Wind {ws} m/s, Humidity {rh}%")
                for k, tmrt in enumerate(tmrts):
                    for l, ta in enumerate(temps):
                        pet_val = calculate_PET_index_vec(ta, rh, tmrt, ws, pet)
                        dset[i, j, k, l] = pet_val

print("database done")