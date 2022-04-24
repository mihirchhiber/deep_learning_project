# Import the required Libraries
from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import os


class Gui():

    def __init__(self):
        self.filepath = ""

        # Create the root window
        self.win = Tk()
        self.win.title('Music Recommender System')
        self.win.geometry("900x300")
        self.win.config(background = "white")

        # Add a Label widget
        label = Label(self.win, text="Input a song (.wav file)", font=('Georgia 13'))
        label.pack(pady=10)

        # Create a Button
        ttk.Button(self.win, text="Browse", command=self.open_file).pack(pady=20)

        self.win.mainloop()

    def open_file(self):
        file = filedialog.askopenfile(mode='r', filetypes=[('Audio Files', '*.wav')])
        if file:
            filepath = os.path.abspath(file.name)
            Label(self.win, text="The File is located at : " + str(filepath), font=('Aerial 11')).pack()
            self.filepath = filepath

if __name__ == "__main__":
    gui = Gui()
    print(gui.filepath)
