from flask import Flask, render_template,request
import pickle
import numpy as np
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/forms.html')
def form():
    return render_template('forms.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/result', methods=['POST','GET'])
def result():
    if request.method == 'POST':
        result1=request.form.to_dict()

        print("hello..................................................................")
        file=open('cancer.pickle', 'rb')
        print("hello1..................................................................")

        instance=pickle.load(file)
        print("hello2..................................................................")

        print("hello3..................................................................")
        d=list(result1.values())
        d1=d[0:9]
        d2=list(np.int_(d1))
        print("hello4..................................................................")
        print(d2)
        prediction=instance.predict([d2])
        print("hello5..................................................................")

        return render_template('result.html',result=prediction)



if __name__ == '__main__':
    app.run(debug=True)
    app.debug(True)
