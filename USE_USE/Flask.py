import traceback
from flask import Flask, request, jsonify
import json
from Model_testing_USE_USE import predict

app = Flask(__name__)

@app.route('/api/classify_products', methods=['POST'])
def article_recommend():
    if request.method == 'POST':
        try:
            data = json.loads(request.data)
            title = data.get('title')
            text = data.get('text')
            response = predict(title=title, text=text)

        except Exception as ex:
            response = {
                'error': 'exception returning categories - {0}'.format(str(traceback.format_exc())),
                'res': []}
    else:
        response = {'res': "request not sent using POST"}


    return jsonify(**{'response':response})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)


