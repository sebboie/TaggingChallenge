"""Module that contains some global variables."""

# Fix a certain height and width, images smaller are excluded and larger images
# are center cropped
HEIGHT = 200
WIDTH = 300

# Fix a random seed for reproducibility
RANDOM_SEED = 2019

# Food tags that have been used at least 800 times.
FOOD_CATEGORIES = sorted([
    "Coffee & Tea", "Sandwiches", "Fast Food", "American", "Pizza", "Burgers",
    "Breakfast & Brunch", "Italian", "Mexican", "Specialty Food", "Chinese",
    "Bakeries", "Cafes", "Desserts", "Ice Cream & Frozen Yogurt", "Japanese",
    "Chicken Wings", "Seafood", "Salad", "Caterers", "Sushi Bars", "Delis",
    "Canadian", "Asian Fusion", "Mediterranean", "Barbeque", "Steakhouses",
    "Indian", "Thai", "Diners", "Vietnamese", "Middle Eastern", "Greek",
    "Vegetarian", "French", "Ethnic Food", "Korean", "Buffets"
])


# Specify the total number of tags/classes
NO_CLASSES = len(FOOD_CATEGORIES)
