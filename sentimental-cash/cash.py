from cs50 import *


def get_cents():
    while True:
        user_cents = get_float("Change owed: ")
        if user_cents >= 0:
            break

    return user_cents * 100

def get_quarters(user_cents):
    quarters = 0
    while user_cents >= 25:
        user_cents -= 25
        quarters += 1
    return quarters

def get_dimes(user_cents):
    dimes = 0
    while user_cents >= 10:
        user_cents -= 10
        dimes += 1
    return dimes

def get_nickels(user_cents):
    nickels = 0
    while user_cents >= 5:
        user_cents -= 5
        nickels += 1
    return nickels

def get_pennies(user_cents):
    pennies = 0
    while user_cents >= 1:
        user_cents -= 1
        pennies += 1
    return pennies

user_cents = get_cents()

quarters = get_quarters(user_cents)
user_cents = user_cents - (quarters * 25)

dimes = get_dimes(user_cents)
user_cents = user_cents - (dimes * 10)

nickels = get_nickels(user_cents)
user_cents = user_cents - (nickels * 5)

pennies = get_pennies(user_cents)


#sum coins

coins = quarters + dimes + nickels + pennies

print(f"{coins}")



