import tkinter as tk
import random

root = tk.Tk()
root.geometry('500x500')
root.title('Roll Dice')

label = tk.Label(root, text='', font=('Times Roman', 100))
roll_label = tk.Label(root, text='', font=('Times Roman', 20))
roll_label1 = tk.Label(root, text='', font=('Times Roman', 20))

def roll_dice():
    dice = ['\u2680', '\u2681', '\u2682', '\u2683', '\u2684', '\u2685']
    d = {'\u2680': 1, '\u2681': 2, '\u2682': 3, '\u2683': 4, '\u2684': 5, '\u2685': 6}
    die1_roll = random.choice(dice)
    die2_roll = random.choice(dice)
    die = [die1_roll, die2_roll]
    die1 = d[die[0]] + d[die[1]]
    label.configure(text=f'{die[0]}{die[1]}')
    label.pack()
    roll_label.configure(text=f'You rolled a {d[die[0]]} and {d[die[1]]}')
    roll_label.pack()

    if die1 == 7 or die1 == 11:
        print(roll_label1.configure(text=f'You Win ' + '\U0001F603 \U0001F389'))
    else:
        print(roll_label1.configure(text=f'Sorry You Lose ' + '\U0001F614 \nTry Again'))

    roll_label1.pack()

button = tk.Button(root, text='Dice Simulator', foreground='green', command=roll_dice)
button.pack()
root.mainloop()
