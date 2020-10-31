import pickle as pkl


with open('/home/dhruv/Documents/python/ai-application/models/sentimentmodel2.pkl','rb') as f:
    sentiment_model = pkl.load(f)

with open('/home/dhruv/Documents/python/ai-application/models/sentimentvc2.pkl','rb') as f:
    sentiment_vectorizer = pkl.load(f)

with open('/home/dhruv/Documents/python/ai-application/models/news_classifier.pkl','rb') as f:
    news_model = pkl.load(f)

with open('/home/dhruv/Documents/python/ai-application/models/news_vectorizer.pkl','rb') as f:
    news_vectorizer = pkl.load(f)

def predict(model, vectorizer, raw):
    x_bar = vectorizer.transform([str(raw)])
    y_bar = model.predict(x_bar)
    return y_bar[0]

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/joinus")
def joinus():
    return render_template("joinus.html")

@app.route("/sentiment_classifier")
def sentiment_classifier():
    return render_template("sentiment.html")

@app.route("/sentiment_classifier", methods=["POST"])
def sentiment_classifier_response():
    text = request.form['text']
    response = predict(sentiment_model, sentiment_vectorizer, text)
    return render_template("sentiment_response.html", value = response)

@app.route("/fake_news_detector")
def fake_news_detector():
    return render_template("fake_news_detector.html")

@app.route("/fake_news_detector", methods=["POST"])
def fake_news_detector_response():
    text = request.form['text']
    response = predict(news_model, news_vectorizer, text)
    return render_template("fake_news_detector_response.html", value = response)

if __name__ == "__main__":
    app.run(debug=True)
