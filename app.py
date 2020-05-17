from flask import Flask, render_template, Response
import flask_socketio
import BVARface
import stream

app = Flask(__name__)
app.config["SECRET_KEY"] = "\x1cSE,@\xd5\x81:7\xda\x91[\xc80 \xea\x14[S~r\xb3v\x1f"
app.config["DEBUG"] = True

socketio = flask_socketio.SocketIO(app)
stream = stream.Video(BVARface.Augment())

@app.route("/")
def index():
    return render_template("index.html")

def spin():
    app.logger.info("Creating stream.")
    while True:
        yield (b"img\r\nContent-Type: img/jpeg\r\n\r\n" + stream.dequeue())

@app.route("/spin")
def output():
    return Response(spin(), mimetype="multipart/x-mixed-replace; boundary=img")

@socketio.on("send")
def test_message(img):
    stream.enqueue(img.split(",")[1])

@socketio.on("conn")
def test_connect():
    app.logger.info("Client has joined session.")

if __name__ == "__main__":
    socketio.run(app, threaded=True)
