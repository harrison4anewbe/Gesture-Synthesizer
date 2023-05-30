import threading
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time
import ktrain
from ktrain import text


# Tooltip class for displaying hints
class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tip_window = None
        self.id = None
        self.x = self.y = 0
        self.text = "Enter an emotion (e.g., Joy, Sad, Anger, Fear, Neutral)"

    def showtip(self):
        "Display text in tooltip window"
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        # Creates a toplevel window
        self.tip_window = tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, background="#ffffe0", relief="solid", borderwidth=1
        )
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


def create_tooltip(widget, text):
    tool_tip = ToolTip(widget)

    def enter(event):
        tool_tip.showtip()

    def leave(event):
        tool_tip.hidetip()

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)


def stop_thread(thread):
    if thread:
        for t in threading.enumerate():
            if t is thread:
                t._stop()


def play_video(video_path, frame, video_label, stop_event):
    def video_loop():
        cap = cv2.VideoCapture(video_path)
        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if ret:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = ImageTk.PhotoImage(image=Image.fromarray(image))
                    video_label.config(image=image)
                    video_label.image = image
                else:
                    break
        finally:
            cap.release()

    video_label.config(image=None)
    video_label.image = None
    thread = threading.Thread(target=video_loop)
    thread.start()
    return thread, stop_event


def gesture_synthesizer(input_word, frame, video_label, current_thread=None):
    video_paths = {
        "joy": "./mov/joy.mov",
        "sad": "./mov/sad.mov",
        "anger": "./mov/joy.mov",
        "fear": "./mov/joy.mov",
        "neutral": "./mov/joy.mov",
    }
    video_path = video_paths.get(input_word.lower())
    if video_path:
        messagebox.showinfo(
            "Gesture synthesizer", f"Generating {input_word} gesture..."
        )
        stop_event = threading.Event()
        if current_thread[0] is not None:  # Check if a thread exists
            current_thread[1].set()  # stop the previous thread
            current_thread[0].join()  # wait for the previous thread to finish
        current_thread = (
            play_video(video_path, frame, video_label, stop_event),
            stop_event,
        )

    return current_thread


def generate_gesture(input_word, frame, video_label, current_thread=None):
    emotions = ["joy", "sad", "anger", "fear", "neutral"]
    if input_word.lower() in emotions:
        gesture_synthesizer(input_word, frame, video_label, current_thread)
    else:
        messagebox.showinfo("Error", "Invalid input. Please enter a valid emotion.")
    return current_thread


def main():
    window = tk.Tk()
    window.title("Gesture Synthesizer")
    window.configure(bg="white")
    window.geometry("800x600")

    frame = tk.Frame(window)
    frame.place(relx=0.5, rely=0.5, anchor="center")

    video_label = tk.Label(frame)
    video_label.pack()

    current_thread = None, None

    input_field = tk.Entry(frame)
    input_field.pack()
    create_tooltip(
        input_field, "Enter an emotion (e.g., joy, sad, anger, fear, neutral)"
    )

    input_button = tk.Button(
        frame,
        text="Generate Gesture",
        command=lambda: generate_gesture(
            input_field.get(), frame, video_label, current_thread
        ),
    )
    input_button.pack()
    message = "Though I do not know how to deal with it, I can still work on it"
    predictor = ktrain.load_predictor("./models/bert_")
    prediction = predictor.predict(message)
    print(prediction)
    window.mainloop()


if __name__ == "__main__":
    main()
