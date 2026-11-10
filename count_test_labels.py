import csv

FILENAME = "NLE_Winners_2004-2025.csv"
TRAIN_END_YEAR = 2016
LOCAL_POSITIONS = ["MAYOR", "VICE MAYOR", "COUNCILOR"]

# Read the CSV and remove dupes
officials = []
seen = set()

with open(FILENAME, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        last_name = row["Last Name"].strip() or "N/A"
        first_name = row["First Name"].strip() or "N/A"
        middle_name = row["Middle Name"].strip() or "N/A"
        sex = row["Sex"].strip() or "N/A"
        position = row["Position"].strip() or "N/A"
        region = row["Region"].strip() or "N/A"
        province = row["Province"].strip() or "N/A"
        city = row["City"].strip() or "N/A"
        year = int(row["Year"]) if row["Year"].strip() else "N/A"

        # Unique key to skip duplicate entries
        key = (last_name, first_name, middle_name, sex, position, region, province, city, year)
        if key not in seen:
            seen.add(key)
            officials.append({
                "last_name": last_name,
                "first_name": first_name,
                "province": province,
                "city": city,
                "position": position,
                "year": year
            })
            
# Keep only local positions with complete information

local_officials = []
for o in officials:
    if (
        o["position"] in LOCAL_POSITIONS
        and o["city"] != "N/A"
        and o["last_name"] != "N/A"
        and o["year"] != "N/A"
    ):
        local_officials.append(o)
        
# Labelling of the dynasties
# Collect all first names that share the same last name in the same city, province, and year

groups = {}
for o in local_officials:
    key = (o["province"], o["city"], o["year"], o["last_name"])
    if key not in groups:
        groups[key] = set()
    groups[key].add(o["first_name"])

# If 2 or more distinct first names share the last name -> Dynasty (1), else -> Not Dynasty (0)
for o in local_officials:
    key = (o["province"], o["city"], o["year"], o["last_name"])
    if len(groups[key]) >= 2:
        o["label"] = 1
    else:
        o["label"] = 0

# Split test set
test_set = []
for o in local_officials:
    if o["year"] > TRAIN_END_YEAR:
        test_set.append(o)
        
# Count the label distrinution in the test set
non_dynasties = 0
dynasties = 0

for o in test_set:
    if o["label"] == 1:
        dynasties += 1
    else:
        non_dynasties += 1

# Printing
print(f"Total test rows: {len(test_set):,}")
print(f"Non-dynasties (Class 0): {non_dynasties:,}")
print(f"Dynasties (Class 1): {dynasties:,}")