import keyboard
import time

while True:
    if keyboard.read_key() == "p":
        print("You pressed p")
    elif keyboard.is_pressed("q"):
        print("You pressed q")