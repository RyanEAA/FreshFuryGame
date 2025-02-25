import random

# gets a random fruit string from the list of fruits
def get_random_fruit() -> str:
    # Create fruit
    fruits = ["Apple.png", "Banana.png", "BellPepper-orange_Half.png", "BellPepper-orange.png", "Jalepeño_Half.png", "Jalepeño.png", "Orange.png", "Tomato_Half.png", "Tomato.png"]
    return "/Users/ryan/Downloads/PFIcons/"+random.choice(fruits)

# def spawn_fruits()