import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import threading
import time
from models.attention_model2 import Detector

videoPath = 0
configPath = os.path.join("server", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
modelPath = os.path.join("server", "frozen_inference_graph.pb")
classesPath = os.path.join("server", "coco.names")
actionModelPath = os.path.join("server", "resnet18.onnx")
actionClassesPath = os.path.join("server", "labels.txt")

class AttentionApp:
    def __init__(self, root, detector):
        self.root = root
        self.detector = detector
        self.attention_scores = {}
        self.lock = threading.Lock()  # Lock to handle thread-safe data updates
        self.running = False  # Control variable for the thread

        # Create a frame for the chart
        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure for the attention chart
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_ylim([0, 10])
        self.ax.set_title('Student Attention Score Over Time')
        self.ax.set_ylabel('Attention Score')
        self.ax.set_xlabel('Time')

        # Create a canvas to show the figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Button to start processing
        self.start_button = tk.Button(self.root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10)

        # Stop button
        self.stop_button = tk.Button(self.root, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack(pady=10)

    def start_detection(self):
        if not self.running:  # Start the detection only if it is not running
            self.running = True
            self.detector_thread = threading.Thread(target=self.run_detection)
            self.detector_thread.start()
            self.update_chart_loop()  # Periodic chart update using Tkinter's after()

    def stop_detection(self):
        self.running = False  # Set the flag to stop the detection
        
        if self.detector_thread.is_alive():
            self.detector_thread.join()  # Wait for the thread to finish

    def run_detection(self):
        while self.running:
            # This is running in a separate thread to prevent UI freezing
            attention_scores = self.detector.get_student_attention()
            
            # Acquire lock before updating shared data structure
            with self.lock:
                self.process_scores(attention_scores)
            
            time.sleep(3)

        if self.running == False:
            self.detector_thread._stop()
            

    def process_scores(self, attention_scores):
        # This function is for thread-safe data processing
        for score in attention_scores:
            student_id = score['student_id']
            if student_id not in self.attention_scores:
                self.attention_scores[student_id] = []
            self.attention_scores[student_id].append(score['attention_score'])

    def update_chart_loop(self):
        # This is scheduled on the Tkinter main loop and will update the chart periodically
        if self.running:
            self.update_chart()
            self.root.after(3000, self.update_chart_loop)

    def update_chart(self):
        # Thread-safe chart update
        self.ax.clear()
        self.ax.set_ylim([0, 10])
        self.ax.set_title('Student Attention Score Over Time')
        self.ax.set_ylabel('Attention Score')
        self.ax.set_xlabel('Time')

        has_data = False  # Flag to check if there's any data to plot

        with self.lock:
            for student_id, scores in self.attention_scores.items():
                if len(scores) > 0:
                    self.ax.plot(scores, label=f'Student {student_id}')
                    has_data = True

        if has_data:
            self.ax.legend()  # Only show the legend if data exists

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Attention Monitoring")

    # Initialize the Detector class
    detector = Detector(
        videoPath=videoPath,
        configPath=configPath,
        modelPath=modelPath,
        classesPath=classesPath,
        actionModelPath=actionModelPath,
        actionClassesPath=actionClassesPath
    )

    app = AttentionApp(root, detector)
    root.mainloop()
