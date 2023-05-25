from flask import Flask, render_template, request

from neural_network import predict

DEBUG = False
app = Flask(__name__)
app.config.from_object(__name__)
app.config["SECRET_KEY"] = "anything"


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        text = request.form["text"].rstrip()
        prediction = predict(text)
        result = "Позитивное сообщение" if prediction else "Негативное сообщение"
        return render_template("main.html", result=result, color="positive" if prediction else "negative", text=text)
    return render_template("main.html")


if __name__ == "__main__":
    app.run()
