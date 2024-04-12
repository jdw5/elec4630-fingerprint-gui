import fingerprints as fp
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
from db import Database
from app import App

    

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
