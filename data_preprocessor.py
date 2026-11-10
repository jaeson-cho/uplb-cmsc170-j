# OpenHalalan Dataset Preprocessor
#   - Cleans Dataset By ONLY Storing Relevant Information Including:
#       1. Personal Info:
#           - Last Name, First Name, Middle Name, Sex.
#       2. Job Description:
#            - Position, Region, Province, City, Year.
# NOTE: Can Be Imported to Main File For Usage.

import csv

def preProcessData(filename):
    officials = []
    unique = set()
    # duplicate_entries = []        # Uncomment to Track Duplicate Entries.
    
    try:
        with open(filename, "r") as file:
            reader = csv.DictReader(file)
            count = 0
            for row_data in reader:
                temp_dict = dict(
                    last_name = row_data["Last Name"].strip() or "N/A",
                    first_name = row_data["First Name"].strip() or "N/A",
                    middle_name = row_data["Middle Name"].strip() or "N/A",
                    sex = row_data["Sex"].strip() or "N/A",
                    position = row_data["Position"].strip() or "N/A",
                    region = row_data["Region"].strip() or "N/A",
                    province = row_data["Province"].strip() or "N/A",
                    city = row_data["City"].strip() or "N/A",
                    year = int(row_data["Year"]) if row_data["Year"].strip() else "N/A"
                )

                # Optimized Approach For Checking Duplicate Entries.
                key = tuple(temp_dict.items())
                if key not in unique:
                    officials.append(temp_dict)
                    unique.add(key)
                # count +=1
                #if count > 100:
                #    break
                # Uncomment Else-Block to Track Duplicate Entries.
                # else:
                #     duplicate_entries.append(temp_dict) 
    except FileNotFoundError:
        print("ERROR: File Not Found!")
        return

    return officials

# Uncomment SCRIPT FOR CHECKING ONLY.
# dataset = preProcessData("NLE_Winners_2004-2025.csv")
# for data in dataset:
#   print(data)