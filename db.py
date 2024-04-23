import sqlite3
from datetime import datetime

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('app.db')
        self.cursor = self.conn.cursor()

        # Create table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS fingerprints (
                id integer PRIMARY KEY,
                name text NOT NULL,
                image_path text NOT NULL,
                template_path text NOT NULL,
                created_at text,
                updated_at text
            );  
        ''')
        self.conn.commit()

    def index(self):
        self.cursor.execute("SELECT * FROM fingerprints")
        rows = self.cursor.fetchall()
        return rows
    
    def count(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM fingerprints")
        row = self.cursor.fetchone()
        return row[0]

    def show(self, id: int):
        self.cursor.execute("SELECT * FROM fingerprints WHERE id=?", (id,))
        row = self.cursor.fetchone()
        return row

    def store(self, name: str, image_path: str, template_path: str):
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_at = created_at
        self.cursor.execute("INSERT INTO fingerprints (name, image_path, template_path, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (name, image_path, template_path, created_at, updated_at))

        self.conn.commit()
        return self.cursor.lastrowid
    
    def destroy(self, id: int):
        self.cursor.execute("DELETE FROM fingerprints WHERE id=?", (id,))
        self.conn.commit()

    # def seed(self):
    
