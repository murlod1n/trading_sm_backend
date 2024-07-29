from flask import Flask, jsonify, request
import chart

app = Flask(__name__)



@app.route('/chart', methods=['GET'])
def get_employees():
    try:
        json = chart.get_chart()

        message = {
            'status': 200,
            'message': 'OK',
            'chart': json
        }
        resp = jsonify(message)
        resp.status_code = 200

        return resp
    except:
        message = {
            'status': 100,
            'message': 'ERROR',
        }
        resp = jsonify(message)
        resp.status_code = 100
        return resp


@app.route("/")
def hello_world():
    return "Hello world"

if __name__ == '__main__':
   app.run(port=5000)