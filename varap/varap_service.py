from flask import Flask
from flask import url_for

app = Flask(__name__)

# simple routing
@app.route('/')
@app.route('/index')
def main_page():
    return 'Video Action Recognition Analyzing Platform(VARAP)'

