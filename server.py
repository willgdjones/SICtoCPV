from flask import Flask, render_template, request

app = Flask(__name__)

from SICtoCPVfunctions import *

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert')
def convert():
    sic_code = request.args.get('SICcode')
    CPVlist = SICtoCPV(sic_code)

    return render_template('convert.html', CPVlist=CPVlist)


if __name__ == '__main__':
    app.run(debug=True)
