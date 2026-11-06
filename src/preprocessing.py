import random
from collections import Counter
from src.data_loader import load_projects


# groups less common categories under "Other" to simplify feature encoding
def group_category(category, common_categories):
    if category in common_categories:
        return category

    return "Other"


# converts project categories into numerical values without assigning them an order
def encode_category(category, common_categories):
    grouped_category = group_category(category, common_categories)
    category_names = common_categories + ["Other"]

    encoded_category = []

    for name in category_names:
        if grouped_category == name:
            encoded_category.append(1)
        else:
            encoded_category.append(0)

    return encoded_category


# converts project regions into numerical values without assigning them an order
def encode_region(region, region_names):
    encoded_region = []

    for name in region_names:
        if region == name:
            encoded_region.append(1)
        else:
            encoded_region.append(0)

    return encoded_region


# scales the infrastructure year between 0 and 1 using the training year range
def scale_year(year, min_year, max_year):
    if min_year == max_year:
        return 0

    return (year - min_year) / (max_year - min_year)


# combines the encoded project features and scaled year into one numerical input
def encode_project(project, common_categories, region_names, min_year, max_year):
    encoded_category = encode_category(project["category"], common_categories)
    encoded_region = encode_region(project["region"], region_names)
    scaled_year = scale_year(project["year"], min_year, max_year)

    return encoded_category + encoded_region + [scaled_year]


# separates the encoded project features and actual budgets for model training and testing
def prepare_features(projects, common_categories, region_names, min_year, max_year):
    features = []
    budgets = []

    for project in projects:
        encoded_project = encode_project(project, common_categories, region_names, min_year, max_year)

        features.append(encoded_project)
        budgets.append(project["budget"])

    return features, budgets


# splits the projects into training and testing datasets
def split_projects(projects):
    shuffled_projects = projects.copy()

    # sets a fixed seed to ensure all algorithms use the same train/test split
    random.seed(67)

    # randomly rearranges the projects to avoid bias from their original order
    random.shuffle(shuffled_projects)

    # allocates 80% for training and 20% for testing
    split_index = int(len(shuffled_projects) * 0.8)

    training_projects = shuffled_projects[:split_index]
    testing_projects = shuffled_projects[split_index:]

    return training_projects, testing_projects


# prepares the same training and testing data for all three algorithms
def prepare_data():
    projects = load_projects()
    training_projects, testing_projects = split_projects(projects)

    # selects the five most common categories using training data only
    category_counts = Counter(project["category"] for project in training_projects)
    common_categories = []

    for category, count in category_counts.most_common(5):
        common_categories.append(category)

    # collects and sorts training regions to keep their encoded positions consistent
    training_regions = set()

    for project in training_projects:
        training_regions.add(project["region"])

    region_names = sorted(training_regions)

    # finds the training year range to avoid using testing data for scaling
    min_year = training_projects[0]["year"]
    max_year = training_projects[0]["year"]

    for project in training_projects:
        if project["year"] < min_year:
            min_year = project["year"]

        if project["year"] > max_year:
            max_year = project["year"]

    # applies the same encoding and scaling rules to both datasets
    X_train, y_train = prepare_features(training_projects, common_categories, region_names, min_year, max_year)
    X_test, y_test = prepare_features(testing_projects, common_categories, region_names, min_year, max_year)

    return X_train, y_train, X_test, y_test


# tests the data splitting
if __name__ == "__main__":
    projects = load_projects()
    training_projects, testing_projects = split_projects(projects)

    print("Training projects:", len(training_projects))
    print("Testing projects:", len(testing_projects))
    
    
    # collects the unique categories and regions from the training data
    training_categories = set()
    training_regions = set()

    for project in training_projects:
        training_categories.add(project["category"])
        training_regions.add(project["region"])

    print("Training categories:", len(training_categories))
    print("Training regions:", len(training_regions))
    
    
    # collects testing categories to check for values not present in training
    testing_categories = set()

    for project in testing_projects:
        testing_categories.add(project["category"])

    # identifies categories that appear only in the testing dataset
    unseen_categories = testing_categories - training_categories

    print("Unseen testing categories:", len(unseen_categories))
    
    
    # collects testing regions to check for values not present in training
    testing_regions = set()

    for project in testing_projects:
        testing_regions.add(project["region"])

    # identifies regions that appear only in the testing dataset
    unseen_regions = testing_regions - training_regions

    print("Unseen testing regions:", len(unseen_regions))
    

    # counts training projects per category to help decide how to encode them
    category_counts = Counter(project["category"] for project in training_projects)

    # displays the 10 most common categories in the training dataset
    for category, count in category_counts.most_common(10):
        print(category, count)
    
    
    # selects the five most common categories from the training data (94.7% of training projects)
    common_categories = []

    for category, count in category_counts.most_common(5):
        common_categories.append(category)

    # checks how less common categories are grouped
    print("Grouped category:", group_category("GAA 2025 SSP", common_categories))

    # checks how common and uncommon categories are encoded
    print("Roads:", encode_category("Roads", common_categories))
    print("GAA 2025 SSP:", encode_category("GAA 2025 SSP", common_categories))
    
    # sorts the regions to keep their positions consistent during encoding
    region_names = sorted(training_regions)

    # checks how a project's region is encoded
    sample_region = training_projects[0]["region"]

    print("Sample region:", sample_region)
    print("Encoded region:", encode_region(sample_region, region_names))

    
    # finds the earliest and latest training years for feature scaling
    min_year = training_projects[0]["year"]
    max_year = training_projects[0]["year"]

    for project in training_projects:
        year = project["year"]

        if year < min_year:
            min_year = year

        if year > max_year:
            max_year = year

    print("Minimum training year:", min_year)
    print("Maximum training year:", max_year)
    
    
    # checks the combined numerical features of one training project
    sample_project = training_projects[0]
    encoded_project = encode_project(sample_project, common_categories, region_names, min_year, max_year)

    print("Encoded project:", encoded_project)
    print("Total features:", len(encoded_project))
    print("Actual budget:", sample_project["budget"])

    # checks the scaled year of our sample project
    print("Original year:", sample_project["year"])
    print("Scaled year:", scale_year(sample_project["year"], min_year, max_year))
    
    # checks if the encoded features and actual budgets are separated correctly
    features, budgets = prepare_features(training_projects[:3], common_categories, region_names, min_year, max_year)

    print("Prepared projects:", len(features))
    print("Features per project:", len(features[0]))
    print("First project's budget:", budgets[0])
    
    
    # prepares the training and testing data using the same feature encoding
    X_train, y_train = prepare_features(training_projects, common_categories, region_names, min_year, max_year)
    X_test, y_test = prepare_features(testing_projects, common_categories, region_names, min_year, max_year)

    # checks the number of prepared inputs and target values
    print("Training inputs:", len(X_train))
    print("Training targets:", len(y_train))
    print("Testing inputs:", len(X_test))
    print("Testing targets:", len(y_test))
