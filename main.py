import os
import sys

sys.path.append(os.path.abspath('003__Face_Recognition_Authentication/Scripts'))

db_path = "003__Face_Recognition_Authentication/Database/face_database.db"

from Scripts.create_db import create_db
from Scripts.register import register_face
from Scripts.login import login_face

if __name__ == "__main__":
    if os.path.exists(db_path):
        register_face('Minnie', 'Minnie', '003__Face_Recognition_Authentication/Minnie01.png', db_path)
        login_face('Minnie', 'Minnie', '003__Face_Recognition_Authentication/Minnie02.png', db_path)
    else: 
        create_db(db_path)
        register_face('Minnie', 'Minnie', '003__Face_Recognition_Authentication/Minnie01.png', db_path)
        login_face('Minnie', 'Minnie', '003__Face_Recognition_Authentication/Minnie02.png', db_path)
