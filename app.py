import fingerprints as fp
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import Scrollbar, Listbox
from db import Database
from storage import Storage
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image

THRESHOLD = 0.75

class App:
    def __init__(self, root):
        # Setup database
        self.db = Database()

        # Hook into storage
        self.storage = Storage()

        self.root = root
        self.root.title("Fingerprint")
        self.root.geometry("600x400")

        # Create layout
        self.sidebar = tk.Frame(root, bg="lightgray", width=150, height=400)
        self.sidebar.pack(side="left", fill="y")

        self.body = tk.Frame(root, bg="white", width=350, height=400, padx=10, pady=10)
        self.body.pack(side="right", fill="both", expand=True)

        self.title = tk.Label(self.sidebar, text="Fingerprint", font=("Arial", 11), bg="lightgray", padx=10, pady=10)
        self.title.pack(side="top")

        # Buttons
        self.enrol_button = tk.Button(self.sidebar, text="Home", command=self.to_home)
        self.enrol_button.pack(side="top", pady=10, padx=10)

        self.enrol_button = tk.Button(self.sidebar, text="Enrol", command=self.to_enrol)
        self.enrol_button.pack(side="top", pady=10, padx=10)

        self.compare_button = tk.Button(self.sidebar, text="Compare", command=self.to_compare)
        self.compare_button.pack(side="top", pady=10, padx=10)

        self.match_button = tk.Button(self.sidebar, text="Match", command=self.to_match)
        self.match_button.pack(side="top", pady=10, padx=10)
        
        self.roc_button = tk.Button(self.sidebar, text="ROC", command=self.to_roc)
        self.roc_button.pack(side="top", pady=10, padx=10)

        # Initial page
        self.current_page = None
        self.to_home()
    
    def page_change(self, page: str):
        self.title.config(text=page)
        if self.current_page:
            self.current_page.destroy()
        self.current_page = tk.Frame(self.body, bg="white", width=350, height=400)
        self.current_page.pack(fill="both", expand=True)

    def to_home(self):
        self.page_change('Home')
        # Add home page content here
        home_label = tk.Label(self.current_page, text="Home Page", font=("Arial", 12), padx=10, pady=10)
        home_label.pack()

        # Add count and display fingerprint details
        count = self.db.count()
        count_label = tk.Label(self.current_page, text=f"Total Fingerprints: {count}", font=("Arial", 10))
        count_label.pack(pady=10)

        grid_frame = tk.Frame(self.current_page)
        grid_frame.pack()

        id_header = tk.Label(grid_frame, text='ID', font=("Arial", 10))
        name_header = tk.Label(grid_frame, text='Name', font=("Arial", 10))
        id_header.grid(row=0, column=0, padx=20)
        name_header.grid(row=0, column=1, padx=10)

        # Display the id and then the name if there are entries
        # When click, show fingerprint
        for i, fingerprint in enumerate(self.db.index()):
            id_label = tk.Label(grid_frame, text=fingerprint[0], font=("Arial", 10))
            name_label = tk.Label(grid_frame, text=fingerprint[1], font=("Arial", 10))
            id_label.grid(row=i+1, column=0, padx=10)
            name_label.grid(row=i+1, column=1, padx=10)

            # Bind the click event to the to_enrol method
            id_label.bind("<Button-1>", lambda event, fp=fingerprint: self.to_show(fp))
            name_label.bind("<Button-1>", lambda event, fp=fingerprint: self.to_show(fp))

    def to_show(self, fingerprint):
        self.page_change('Show')

        name_label = tk.Label(self.current_page, text=f"Name: {fingerprint[1]}", font=("Arial", 10))
        name_label.pack(pady=10)

        # Load the image
        image_path = fingerprint[2]
        image = Image.open(image_path)

        

        # Resize the image if it's too large
        max_size = (600, 600)
        image.thumbnail(max_size)

        image = image.convert('L')

        # Create a new matplotlib figure and draw the image on it
        fig = Figure(figsize=(5, 5), dpi=100)
        a = fig.add_subplot(111)

        a.imshow(image, cmap='gray')
        a.axis('off')  # Hide the axis

        # Create a new FigureCanvasTkAgg object and attach the matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master=self.current_page)  # A tk.DrawingArea.
        canvas.draw()

        # Add the canvas to the Tkinter frame
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def to_enrol(self):
        self.page_change('Enrol')

        # Add enrol page content here
        # Input for name
        name_label = tk.Label(self.current_page, text="Name:", font=("Arial", 10))
        name_label.pack(pady=(10, 5))
        self.name_entry = tk.Entry(self.current_page)
        self.name_entry.pack(pady=5)

        # File upload field
        file_label = tk.Label(self.current_page, text="Upload Fingerprint Image:", font=("Arial", 10))
        file_label.pack(pady=(10, 5))
        self.file_button = tk.Button(self.current_page, text="Choose File", command=self.choose_file)
        self.file_button.pack(pady=5)
        self.file_path_label = tk.Label(self.current_page, text="", font=("Arial", 8))
        self.file_path_label.pack()

        # Submit button
        submit_button = tk.Button(self.current_page, text="Submit", command=self.create_fingerprint)
        submit_button.pack(pady=10)

    def to_compare(self):
        self.page_change('Compare')
        # Add compare page content here
        compare_label = tk.Label(self.current_page, text="Compare Page", font=("Arial", 12), padx=10, pady=10)
        compare_label.pack()

        # File upload field
        file_label = tk.Label(self.current_page, text="Upload image to compare", font=("Arial", 10))
        file_label.pack(padx=10, pady=(10, 5))
        self.file_button = tk.Button(self.current_page, text="Choose File", command=self.choose_file)
        self.file_button.pack(padx=10, pady=5)
        self.file_path_label = tk.Label(self.current_page, text="", font=("Arial", 8))
        self.file_path_label.pack(padx=10)


        listbox_label = tk.Label(self.current_page, text="Select a fingerprint to compare to", font=("Arial", 10), padx=10, pady=10)
        listbox_label.pack()

        # Scrollable list of fingerprints
        listbox_frame = tk.Frame(self.current_page)  # New frame for the listbox and scrollbar
        listbox_frame.pack(fill="x")

        scrollbar = Scrollbar(listbox_frame)
        scrollbar.pack(side="right", fill="y")

        self.available_fingerprints = self.db.index()
        self.fingerprint_list = Listbox(listbox_frame, yscrollcommand=scrollbar.set)
        for fingerprint in self.available_fingerprints:
            self.fingerprint_list.insert(tk.END, f"ID: {fingerprint[0]}, Name: {fingerprint[1]}")
        self.fingerprint_list.pack(side="left", fill="both", expand=True)

        scrollbar.config(command=self.fingerprint_list.yview)

        # Compare button
        compare_button = tk.Button(self.current_page, text="Compare", command=self.compare_fingerprint)
        compare_button.pack(padx=10, pady=10, fill='x')
    
    def to_match(self):
        self.page_change('Match')

        # File upload field
        file_label = tk.Label(self.current_page, text="Upload image to compare", font=("Arial", 10))
        file_label.pack(padx=10, pady=(10, 5))
        self.file_button = tk.Button(self.current_page, text="Choose File", command=self.choose_file)
        self.file_button.pack(padx=10, pady=5)
        self.file_path_label = tk.Label(self.current_page, text="", font=("Arial", 8))
        self.file_path_label.pack(padx=10)

        # Compare button
        compare_button = tk.Button(self.current_page, text="Match", command=self.match)
        compare_button.pack(padx=10, pady=10, fill='x')

    def match(self):
        if not self.file_path_label.cget("text") :
            messagebox.showerror("Error", "Please choose a file.")
            return False
    
        # Process the uploaded image
        try :
            res = fp.pipeline(self.file_path_label.cget("text"))
            f1, m1, ls1 = res
        except: # If an error occurs, display an error dialog
            messagebox.showerror("Error", "An error occurred while processing the uploaded fingerprint.")
            return False
        
        # Get all fingerprints
        fingerprints = [print[3] for print in self.db.index()]

        for i in range(len(fingerprints)):
            current = fingerprints[i]
            current = self.storage.load_npz(current)
            _, ls2 = current
            score = fp.similarity(ls1, ls2)
            if score > THRESHOLD:
                self.result_label = tk.Label(self.current_page, text=f"Match found: {self.db.index()[i][1]}", font=("Arial", 10))
                self.result_label.pack(pady=10)
                return True
            
        self.result_label = tk.Label(self.current_page, text="No match found", font=("Arial", 10))
        self.result_label.pack(pady=10)
        return True


    def compare_fingerprint(self):
        # If there's no image uploaded OR nothing selected, display error dialog
        if not self.file_path_label.cget("text") :
            messagebox.showerror("Error", "Please choose a file.")
            return False
        
        if not self.fingerprint_list.curselection():
            messagebox.showerror("Error", "Please select a fingerprint to compare to.")
            return False
        
        selected_index = self.fingerprint_list.curselection()[0]  # Get the selected index
        selected = self.available_fingerprints[selected_index]

        # Process the uploaded image
        try :
            res = fp.pipeline(self.file_path_label.cget("text"))
            f1, m1, ls1 = res
        except: # If an error occurs, display an error dialog
            messagebox.showerror("Error", "An error occurred while processing the uploaded fingerprint.")
            return False   
        
        # Load the selected fingerprint
        try:
            res = self.storage.load_npz(selected[3])
            m2, ls2 = res
        except:
            messagebox.showerror("Error", "An error occurred while loading the selected fingerprint.")
            return False
        
        # Compare the fingerprints
        try:
            score = fp.similarity(ls1, ls2)
        except:
            messagebox.showerror("Error", "An error occurred while comparing the fingerprints.")
            return False

        # Display the result
        self.result_label = tk.Label(self.current_page, text=f"Similarity score: {round(score * 100, 2)}", font=("Arial", 10))
        self.result_label.pack(pady=10)
        return True
        
    
    def to_roc(self):
        self.page_change('ROC')
        # Add ROC page content here
        roc_label = tk.Label(self.current_page, text="ROC Page", font=("Arial", 12), padx=10, pady=10)
        roc_label.pack()
        
        # Load a large number of fingerprints
        fingerprints = [print[3] for print in self.db.index()]

        if len(fingerprints) < 2:
            messagebox.showerror("Error", "There are not enough fingerprints to compare.")
            return False

        scores = []
        true_matches = []
        for i in range(len(fingerprints)):
            current = fingerprints[i]
            current = self.storage.load_npz(current)
            _, ls1 = current
            for j in range(i, len(fingerprints)):
                
                other = fingerprints[j]
                other = self.storage.load_npz(other)
                _, ls2 = other
                score = fp.similarity(ls1, ls2)
                scores.append(score > THRESHOLD)
                if i == j:
                    true_matches.append(True)
                else:
                    true_matches.append(False)

        if not any(true_matches):
            messagebox.showerror("Error", "There are no true matches, unable to produce ROC curve.")
            return False
                
        fpr, tpr, thresholds = roc_curve(true_matches, scores)
        roc_auc = auc(fpr, tpr)


        fig = Figure(figsize=(5, 5), dpi=100)
        a = fig.add_subplot(111)
        a.plot(fpr, tpr, label='ROC curve')
        a.plot([0, 1], [0, 1], 'k--')
        a.set_xlim([0.0, 1.0])
        a.set_ylim([0.0, 1.05])
        a.set_xlabel('False Positive Rate')
        a.set_ylabel('True Positive Rate')
        a.set_title('Receiver Operating Characteristic')
        a.legend(loc="lower right")

        fnr = 1 - tpr
        closest = np.argmin(np.abs(fnr - 0.01))
        estimated_fpr = fpr[closest]

        # Put a label for it in TK
        roc_label = tk.Label(self.current_page, text=f"FPR for FNR 1%: {estimated_fpr}", font=("Arial", 10))
        roc_label.pack(pady=10)
        # print(f'Estimated FPR for FNR of 1%: {estimated_fpr}')
        
        # Create a new FigureCanvasTkAgg object and attach the matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master=self.current_page)  # A tk.DrawingArea.
        canvas.draw()

        # Add the canvas to the Tkinter frame
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
    def create_fingerprint(self) -> bool:
        name = self.name_entry.get()
        file_path = self.file_path_label.cget("text")
        if not name or not file_path:
            messagebox.showerror("Error", "Please enter a name and choose a file.")
            return False
        
        if not file_path.lower().endswith(('.jpg', '.jpeg', '.png', 'tif')):
            messagebox.showerror("Error", "Please choose a JPG, JPEG, or PNG file.")
            return False
        
        self.processing_label = tk.Label(self.current_page, text="Processing...", font=("Arial", 10))
        self.processing_label.pack()
        try :
            res = fp.pipeline(file_path)
            
            
            try:
                f1, m1, ls1 = res
                uuid_name = self.storage.generate_uuid_name()
                image_path = self.storage.save_image(f1, uuid_name)
                template_path = self.storage.save_npz((m1, ls1), uuid_name)
                self.db.store(name, image_path, template_path)
            except:
                messagebox.showerror("Error", "An error occurred saving the fingerprint data.")
                self.processing_label.config(text="Processing failed.")
                return False
        except:
            messagebox.showerror("Error", "An error occurred while processing the fingerprint.")
            self.processing_label.config(text="Processing failed.")
            return False
        
        self.processing_label.config(text="Processing... Done!")
        return True
        
    def choose_file(self):
        file_path = filedialog.askopenfilename()
        self.file_path_label.config(text=file_path)

