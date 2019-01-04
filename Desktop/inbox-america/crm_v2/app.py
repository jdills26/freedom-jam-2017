# app.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///inbox_db.db'
app.secret_key = "flask rocks!"

db = SQLAlchemy(app)