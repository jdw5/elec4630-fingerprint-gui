import fingerprints as fp
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
from db import Database

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerprint Comparison")
        self.db = Database()

        # Buttons
        self.enrol_button = tk.Button(root, text="Enrol", command=self.open_enrol_ui)
        self.enrol_button.pack(side=tk.LEFT)
        self.compare_button = tk.Button(root, text="Compare", command=self.open_compare_ui)
        self.compare_button.pack(side=tk.LEFT)
        self.stats_button = tk.Button(root, text="Stats", command=self.open_stats_ui)
        self.stats_button.pack(side=tk.LEFT)
        self.roc_button = tk.Button(root, text="ROC", command=self.open_roc_ui)
        self.roc_button.pack(side=tk.LEFT)
        self.view_button = tk.Button(root, text="View", command=self.open_view_ui)
        self.view_button.pack(side=tk.LEFT)
    
    def open_enrol_ui(self):
        self.root.withdraw()  # Hide main window
        enrol_window = tk.Toplevel(self.root)
        enrol_window.title("Enrol UI")
        
        # Back Button
        back_button = tk.Button(enrol_window, text="Back", command=lambda: self.back_to_main(enrol_window))
        back_button.pack()

    def open_compare_ui(self):
        self.root.withdraw()  # Hide main window
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Compare UI")
        
        # Back Button
        back_button = tk.Button(compare_window, text="Back", command=lambda: self.back_to_main(compare_window))
        back_button.pack()

    def open_stats_ui(self):
        self.root.withdraw()  # Hide main window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Stats UI")
        
        # Back Button
        back_button = tk.Button(stats_window, text="Back", command=lambda: self.back_to_main(stats_window))
        back_button.pack()

    def open_roc_ui(self):
        self.root.withdraw()  # Hide main window
        roc_window = tk.Toplevel(self.root)
        roc_window.title("ROC UI")
        
        # Back Button
        back_button = tk.Button(roc_window, text="Back", command=lambda: self.back_to_main(roc_window))
        back_button.pack()

    def open_view_ui(self):
        self.root.withdraw()  # Hide main window
        self.root.geometry("800x600")
        view_window = tk.Toplevel(self.root)
        view_window.title("View UI")
        
        # Back Button
        back_button = tk.Button(view_window, text="Back", command=lambda: self.back_to_main(view_window))
        back_button.pack()

    def back_to_main(self, window):
        window.destroy()
        self.root.deiconify()

    

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = App(root)
    root.mainloop()