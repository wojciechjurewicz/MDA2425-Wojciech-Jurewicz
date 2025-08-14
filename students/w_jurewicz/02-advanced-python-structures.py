# Exercise 1:

my_tuple = (5, "hello", 3.14, "world", 7)
odd_tuple = tuple(sorted(my_tuple[0::2]))
even_tuple = tuple(my_tuple[1::2])

print("%s %s" % even_tuple)
print("{} {}".format(even_tuple[0], even_tuple[1]))
print(f"{even_tuple[0]} {even_tuple[1]}")

# Exercise 2:

my_list = [10, 20, 30, "python", 2.5]
print(my_list[-1])
my_list[1] = 50
for i in my_list:
    print(i)

# Exercise 3:

my_dict = {"name": "Wojciech",
          "age": 20,
          "city": "Lodz"}

print(my_dict["age"])
my_dict["age"] = 13
my_dict["gender"] = "M"

# Exercise 4:

occupation = "student"
name_age = "My name is %s and I am %d years old." % (my_dict["name"], my_dict["age"])
name_occupation = "My name is {name} and I am a {occupation}.".format(name=my_dict["name"], occupation=occupation)
name_age_occupation = f"My name is {my_dict['name']} and I am {my_dict['age']} years old. I am a {occupation}."

# Exercise 5:

def add_numbers(a=1,b=2):
    return a+b

def print_info(**kwargs):
    for _, value in kwargs.items():
        print(value)

# Exercise 6:

def calculate_area(length: float, width: float) -> float:
    return length*width

# Exercise 7:

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**(1/2)
    
# Exercise 8:

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**(1/2)
    
    def __mul__(self, scalar):
        self.x *= scalar
        self.y *= scalar

# Exercise 9:

class CustomException(Exception):
    pass

def division(a,b):
    try:
        print("Division result is: ", a/b)
    except ZeroDivisionError:
        print("Divison impossible")
    else:
        if a == 123:
            raise CustomException

# Exercise 10:

import random

def generate_random_integer():
    return random.randint(1, 100)

def generate_random_float():
    return random.random()

def generate_random_choice(lst):
    if len(lst) >= 5:
        return random.choice(lst)
    return None

if __name__ == "__main__":
    random_int = generate_random_integer()
    random_float = generate_random_float()
    random_choice = generate_random_choice(['a', 'b', ('c', 'd'), 3, 0.001])

    print(f"Random int: {random_int}\nRandom float: {random_float}\nRandom element from a list: {random_choice}")