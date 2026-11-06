import csv

# loads and prepares the project records from the dataset
def load_projects():
    # opens the dataset
    with open("data/dpwh_transparency_data.csv", "r", encoding="utf-8-sig", newline="") as file:
        # reads each row as a dictionary using the column names
        reader = csv.DictReader(file)

        # counts the projects and checks for zero budgets, missing regions
        total_projects = 0
        zero_budgets = 0
        
        missing_regions = 0
        regions = set()         # stores unique values, so duplicate region names won't be counted twice
        
        central_office_projects = 0         # to separate from other regions
        
        categories = set()          # stores distinct project categories
        
        valid_projects = []
        
        invalid_years = 0
        
        for row in reader:        
            total_projects += 1

            if float(row["budget"]) == 0:
                zero_budgets += 1            
            
            # extracts the region and checks for missing values
            region = ""
            location = row["location"]

            if "'region':" in location:
                region = location.split("'region':")[1].replace("}", "").strip()

                if not region:          # if there's no value under region (empty)
                    missing_regions += 1
                else:
                    regions.add(region)
                    # counts projects assigned to the central office
                    if region == "Central Office":
                        central_office_projects += 1
            else:
                missing_regions += 1            # if region is not found at all in location
                
            
            # PROJECT CATEGORIES
            category = row["category"].strip()

            if category:
                categories.add(category)
            
            
            # checks if the infrastructure year is a valid integer
            year = None
            
            try:
                year = int(row["infraYear"])
            except ValueError:
                invalid_years += 1
            
            
            # VALID PROJECTS
            if float(row["budget"]) > 0 and category and region and year is not None:
                valid_projects.append({
                    "category": category,
                    "year": year,
                    "region": region,
                    "budget": float(row["budget"])
                })


    # FOR DEBUGGING 
    
    # print("Total projects:", total_projects)
    # print("Zero-budget projects:", zero_budgets)
    # print("Projects with missing regions:", missing_regions)

    # # displays the unique regions in the dataset
    # print("Unique regions:", len(regions))

    # for region in sorted(regions):
    #     print(region)

    # print("Central Office projects:", central_office_projects)
    # print("Unique categories:", len(categories))
    # print("Valid projects:", len(valid_projects))
    # print("Projects with invalid years:", invalid_years)

    # # displays a sample valid project and checks its data types
    # print("Sample project:", valid_projects[0])
    # print("Year type:", type(valid_projects[0]["year"]))
    # print("Budget type:", type(valid_projects[0]["budget"]))

    return valid_projects

# tests the data loader when this file is run directly
if __name__ == "__main__":
    projects = load_projects()
    print("Returned projects:", len(projects))