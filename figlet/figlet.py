import sys
from pyfiglet import Figlet
import random

figlet = Figlet()
figlet.getFonts()


if len(sys.argv) == 1:
    employRandomFont = True

elif len(sys.argv) == 3 and (sys.argv[1] == "-f" or sys.argv[1] == "--font"):
    employRandomFont = False
else:
    print("Invalid usage")
    sys.exit(1)



if employRandomFont == False:
    try:
        figlet.setFont(font=sys.argv[2])
    except:
        print("Invalid usage")
        sys.exit(1)

else:
    font = random.choice(figlet.getFonts())


user_message = input("Input: ")


print(f"Output: ")
print(figlet.renderText(user_message))