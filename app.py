import fingerprints as fp
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
from db import Database
from storage import Storage

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
    
    def to_roc(self):
        self.page_change('ROC')
        # Add ROC page content here
        roc_label = tk.Label(self.current_page, text="ROC Page", font=("Arial", 12), padx=10, pady=10)
        roc_label.pack()
        
    def create_fingerprint(self) -> bool:
        name = self.name_entry.get()
        file_path = self.file_path_label.cget("text")
        if not name or not file_path:
            messagebox.showerror("Error", "Please enter a name and choose a file.")
            return False
        
        if not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            messagebox.showerror("Error", "Please choose a JPG, JPEG, or PNG file.")
            return False
        
        self.processing_label = tk.Label(self.current_page, text="Processing...", font=("Arial", 10))
        self.processing_label.pack()
        try :
            res = fp.pipeline(file_path)
            
            
            f1, m1, ls1 = res
            # Create a new record in the database
            name = self.storage.generate_uuid_name()
            image_path = self.storage.save_image(f1, name)
            template_path = self.storage.save_npz((m1, ls1), name)
            try:
                print(image_path, template_path)
                self.db.store(name, image_path, template_path)
            except:
                messagebox.showerror("Error", "A database issue has occurred")
                self.processing_label.config(text="Processing failed.")
                return False
        except:
            messagebox.showerror("Error", "An error occurred while processing the fingerprint.")
            self.processing_label.config(text="Processing failed.")
            return False
        
        print(res)
        self.processing_label.config(text="Processing... Done!")
        return True
        

    def choose_file(self):
        file_path = filedialog.askopenfilename()
        self.file_path_label.config(text=file_path)

