import flask

app = flask.Flask(__name__)

# simple routing
@app.route('/')
@app.route('/index')
def main_page():
    return flask.render_template('index.html')


@app.route('/hello')
def hello_template():
    return flask.render_template('hello_world.html', name='MSCV')

