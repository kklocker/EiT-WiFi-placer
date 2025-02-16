#!/usr/bin/env python3
## Stdlib imports ##
import random
import os
import atexit
from threading import Thread
from time import sleep
from functools import partial

# Want to see a ghetto database? _This_ is a ghetto database
import shelve

# For url quoting
import urllib

# For request validation
import re

## External package imports ##
from flask import Flask, render_template, request, flash, redirect, jsonify
from matplotlib.pyplot import imread

## Local imports ##
from RouterPlacer import RouterPlacer


def process_image(key):
    """This actually processes and moves the image"""
    img = imread(f"uploads/{key}.png")
    rp = RouterPlacer(img)
    rp.get_optimal_solution()
    rp.solution_plot(save_path=f"static/results/{key}.png")
    os.remove(f"uploads/{key}.png")


def process_queue():
    """This function runs in a separate thread and runs through the queue and
    tries to empty it. We don't have a way to process the image right now,
    though."""
    while True:
        # Process and pop in two different transactions, so that things can
        # happen while processing.
        with shelve.open("wifi") as db:
            queue = db["queue"]

        if len(queue) > 0:
            key = queue[0]
            process_image(key)
            with shelve.open("wifi", writeback=True) as db:
                db["queue"].pop(0)
        sleep(2)


def randstr(*, length=64):
    chars = "abcdefghijklmnopqrstuvwxyz123456789"
    return "".join(random.choice(chars) for i in range(64))


### Init the db
with shelve.open("wifi", writeback=True) as db:
    if "queue_size" not in db:
        db["queue_size"] = 5

    if "queue" not in db:
        db["queue"] = []

flask = Flask(__name__)

# Flask configuration
flask.secret_key = randstr()


@flask.route("/waitingroom/<key>")
def waitingroom(key):
    if not re.search(r"[a-z0-9]{64}", key):
        flash("Invalid key", "danger")
        return redirect("/")

    if key + ".png" in os.listdir(path="static/results"):
        if "json" in request.args:
            return jsonify(dict(key=key, done=True))
        else:
            return render_template("waitingroom.html", key=key, queue=None)

    with shelve.open("wifi") as db:
        queue = db["queue"]

        if key not in queue:
            flash("Key not in queue", "danger")
            return redirect("/")

        queue_size = db["queue_size"]
        current_position = queue.index(key)

    if "json" in request.args:
        return jsonify(
            dict(
                key=key,
                queue_elements=len(queue),
                queue_size=queue_size,
                current_position=current_position,
                done=False,
            )
        )
    else:
        return render_template(
            "waitingroom.html",
            key=key,
            queue=queue,
            queue_size=queue_size,
            current_position=current_position,
        )


@flask.route("/", methods=("get", "post"))
def index():
    """The index page. Give the client a key to identify them and their
    progress. Show an upload page"""

    # Check if we can upload anything
    with shelve.open("wifi") as db:
        queue = db["queue"]
        queue_size = db["queue_size"]
        queue_full = len(db["queue"]) >= db["queue_size"]

    key = randstr()

    if request.method == "POST":
        # If the request already has a key, keep it
        if "key" in request.form:
            key = request.form.get("key")
            if not re.search(r"[a-z0-9]{64}", key):
                flash("Ugyldig nøkkel", "danger")
                return redirect(request.url)

        if queue_full:
            flash("Køen er full", "danger")
            return redirect(request.url)

        if "floorPlan" not in request.files:
            flash("Ingen fil mottatt", "danger")
            return redirect(request.url)

        file = request.files["floorPlan"]
        if not file.filename.endswith(".png"):
            flash("Feil filtype, bruk png", "danger")
            return redirect(request.url)

        file.seek(0, os.SEEK_END)
        max_size = 2
        if file.tell() > max_size * 10e5:
            flash(f"Filen er for stor, mindre enn {max_size}mb, takk", "danger")
            return redirect("/")
        file.seek(0)

        fname = os.path.join("uploads", randstr() + ".png")
        # Add the file to the queue
        with shelve.open("wifi", writeback=True) as db:
            if key in db["queue"]:
                flash("Denne sesjonen prosesserer allerede et bilde", "danger")
                return redirect("/")

            db["queue"].append(key)

        file.save(f"uploads/{key}.png")
        flash("Filen er lagt til i køen", "success")
        return redirect("/waitingroom/" + key)

    return render_template("index.html", key=key, queue_full=queue_full)


if __name__ == "__main__":
    t = Thread(target=process_queue, daemon=True)
    t.start()
    flask.run(host="0.0.0.0", port=9001, debug=False, use_reloader=False)
