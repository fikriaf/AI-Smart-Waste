import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import queue


class RealTimeTrashDetectorTkinter:
    def __init__(self, root):
        """
        Real-time trash detector menggunakan Tkinter GUI
        """
        self.root = root
        self.root.title("🗑 Real-Time Trash Detector")
        self.root.geometry("1200x800")

        # Variables
        self.model = None
        self.model_path = tk.StringVar(value="model_baru.pt")
        self.confidence_threshold = tk.DoubleVar(value=0.3)
        self.camera_id = tk.IntVar(value=0)
        self.running = False

        # Class names dan colors
        self.class_names = ["botol_kaca", "botol_kaleng", "botol_plastik"]
        self.colors = {
            "botol_kaca": (255, 0, 0),  # Merah (RGB)
            "botol_kaleng": (0, 255, 0),  # Hijau
            "botol_plastik": (0, 0, 255),  # Biru
        }

        # Statistics
        self.total_detections = tk.IntVar(value=0)
        self.class_counts = {
            "botol_kaca": tk.IntVar(value=0),
            "botol_kaleng": tk.IntVar(value=0),
            "botol_plastik": tk.IntVar(value=0),
        }

        # Threading
        self.frame_queue = queue.Queue(maxsize=5)
        self.cap = None

        # Create output directories
        os.makedirs("realtime_captures", exist_ok=True)
        os.makedirs("realtime_logs", exist_ok=True)

        # FPS tracking
        self.fps_counter = 0
        self.fps_start = datetime.now()
        self.current_fps = tk.StringVar(value="0.0")

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        """Setup GUI layout"""

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Control Panel (Left)
        self.setup_control_panel(main_frame)

        # Video Display (Center)
        self.setup_video_display(main_frame)

        # Statistics Panel (Right)
        self.setup_statistics_panel(main_frame)

    def setup_control_panel(self, parent):
        """Setup control panel"""
        control_frame = ttk.LabelFrame(parent, text="🎛 Controls", padding="10")
        control_frame.grid(
            row=0, column=0, rowspan=2, sticky=(tk.N, tk.S, tk.W), padx=(0, 10)
        )

        # Model selection
        ttk.Label(control_frame, text="Model Path:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        model_frame = ttk.Frame(control_frame)
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)

        ttk.Entry(model_frame, textvariable=self.model_path, width=20).grid(
            row=0, column=0, sticky=(tk.W, tk.E)
        )
        ttk.Button(model_frame, text="Browse", command=self.browse_model, width=8).grid(
            row=0, column=1, padx=(5, 0)
        )

        # Camera ID
        ttk.Label(control_frame, text="Camera ID:").grid(
            row=2, column=0, sticky=tk.W, pady=(10, 2)
        )
        ttk.Spinbox(
            control_frame, from_=0, to=5, textvariable=self.camera_id, width=20
        ).grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)

        # Confidence threshold
        ttk.Label(control_frame, text="Confidence:").grid(
            row=4, column=0, sticky=tk.W, pady=(10, 2)
        )
        confidence_frame = ttk.Frame(control_frame)
        confidence_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=2)

        ttk.Scale(
            confidence_frame,
            from_=0.1,
            to=0.9,
            variable=self.confidence_threshold,
            orient=tk.HORIZONTAL,
        ).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Label(confidence_frame, textvariable=self.confidence_threshold).grid(
            row=0, column=1, padx=(5, 0)
        )

        # Buttons
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(
            row=6, column=0, sticky=(tk.W, tk.E), pady=10
        )

        self.load_btn = ttk.Button(
            control_frame, text="📄 Load Model", command=self.load_model
        )
        self.load_btn.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=2)

        self.start_btn = ttk.Button(
            control_frame, text="▶ Start Detection", command=self.start_detection
        )
        self.start_btn.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=2)

        self.stop_btn = ttk.Button(
            control_frame,
            text="⏹ Stop Detection",
            command=self.stop_detection,
            state=tk.DISABLED,
        )
        self.stop_btn.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=2)

        self.capture_btn = ttk.Button(
            control_frame,
            text="📸 Capture",
            command=self.capture_frame,
            state=tk.DISABLED,
        )
        self.capture_btn.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=2)

        self.reset_btn = ttk.Button(
            control_frame, text="🔄 Reset Stats", command=self.reset_statistics
        )
        self.reset_btn.grid(row=11, column=0, sticky=(tk.W, tk.E), pady=2)

    def setup_video_display(self, parent):
        """Setup video display area"""
        video_frame = ttk.LabelFrame(parent, text="📹 Live Detection", padding="10")
        video_frame.grid(
            row=0, column=1, rowspan=2, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5
        )

        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, width=640, height=640, bg="black")
        self.video_canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        # Status bar
        status_frame = ttk.Frame(video_frame)
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(status_frame, text="FPS:").grid(row=0, column=0)
        ttk.Label(status_frame, textvariable=self.current_fps).grid(
            row=0, column=1, padx=(5, 20)
        )

        ttk.Label(status_frame, text="Status:").grid(row=0, column=2)
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="green")
        self.status_label.grid(row=0, column=3, padx=(5, 0))

    def setup_statistics_panel(self, parent):
        """Setup statistics panel"""
        stats_frame = ttk.LabelFrame(parent, text="📊 Statistics", padding="10")
        stats_frame.grid(
            row=0, column=2, rowspan=2, sticky=(tk.N, tk.S, tk.E), padx=(10, 0)
        )

        # Total detections
        ttk.Label(
            stats_frame, text="Total Detections:", font=("Arial", 10, "bold")
        ).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(
            stats_frame,
            textvariable=self.total_detections,
            font=("Arial", 12, "bold"),
            foreground="blue",
        ).grid(row=0, column=1, sticky=tk.E, pady=2)

        ttk.Separator(stats_frame, orient=tk.HORIZONTAL).grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )

        # Class counts
        row = 2
        for class_name, color in self.colors.items():
            # Color indicator
            color_canvas = tk.Canvas(stats_frame, width=20, height=20)
            color_canvas.grid(row=row, column=0, sticky=tk.W, pady=2)
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            color_canvas.create_rectangle(2, 2, 18, 18, fill=color_hex, outline="black")

            # Class name and count
            ttk.Label(stats_frame, text=f"{class_name}:").grid(
                row=row, column=1, sticky=tk.W, padx=(5, 0), pady=2
            )
            ttk.Label(
                stats_frame,
                textvariable=self.class_counts[class_name],
                font=("Arial", 10, "bold"),
            ).grid(row=row, column=2, sticky=tk.E, pady=2)
            row += 1

        ttk.Separator(stats_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )

        # Recent detections
        ttk.Label(
            stats_frame, text="Recent Detections:", font=("Arial", 10, "bold")
        ).grid(row=row + 1, column=0, columnspan=3, sticky=tk.W, pady=2)

        # Listbox for recent detections
        self.recent_listbox = tk.Listbox(stats_frame, height=6, width=25)
        self.recent_listbox.grid(
            row=row + 2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2
        )

        # Scrollbar for recent detections
        scrollbar = ttk.Scrollbar(
            stats_frame, orient=tk.VERTICAL, command=self.recent_listbox.yview
        )
        scrollbar.grid(row=row + 2, column=3, sticky=(tk.N, tk.S))
        self.recent_listbox.configure(yscrollcommand=scrollbar.set)

        # Recent coordinates
        ttk.Label(
            stats_frame, text="Recent Coordinates:", font=("Arial", 10, "bold")
        ).grid(row=row + 3, column=0, columnspan=3, sticky=tk.W, pady=(10, 2))

        # Listbox for recent coordinates
        self.coords_listbox = tk.Listbox(stats_frame, height=6, width=25)
        self.coords_listbox.grid(
            row=row + 4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2
        )

        # Scrollbar for coordinates
        coords_scrollbar = ttk.Scrollbar(
            stats_frame, orient=tk.VERTICAL, command=self.coords_listbox.yview
        )
        coords_scrollbar.grid(row=row + 4, column=3, sticky=(tk.N, tk.S))
        self.coords_listbox.configure(yscrollcommand=coords_scrollbar.set)

    def browse_model(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", ".pt"), ("All Files", ".*")],
        )
        if filename:
            self.model_path.set(filename)

    def load_model(self):
        """Load YOLO model"""
        try:
            model_path = self.model_path.get()
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return

            self.model = YOLO(model_path)
            self.status_label.config(text="Model Loaded", foreground="green")
            messagebox.showinfo("Success", f"Model loaded successfully!\n{model_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.status_label.config(text="Model Load Failed", foreground="red")

    def detect_objects(self, frame):
        """Deteksi objek pada frame"""
        if self.model is None:
            return []

        try:
            results = self.model(frame, verbose=False)
            detections = []

            if results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    if confidence >= self.confidence_threshold.get() and class_id < len(
                        self.class_names
                    ):
                        class_name = self.class_names[class_id]
                        bbox = box.xyxy[0].tolist()

                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)

                        detection = {
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": [int(coord) for coord in bbox],
                            "center": [center_x, center_y],
                        }
                        detections.append(detection)

            return detections

        except Exception as e:
            print(f"❌ Error in detection: {e}")
            return []

    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            center_x, center_y = det["center"]
            class_name = det["class"]
            confidence = det["confidence"]

            # Get color (convert RGB to BGR for OpenCV)
            color_rgb = self.colors.get(class_name, (128, 128, 128))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)

            # Draw center point
            cv2.circle(annotated_frame, (center_x, center_y), 6, color_bgr, -1)
            cv2.circle(annotated_frame, (center_x, center_y), 8, (255, 255, 255), 2)

            # Label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - 30),
                (x1 + label_size[0] + 10, y1),
                color_bgr,
                -1,
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Coordinates
            coord_text = f"({center_x},{center_y})"
            cv2.putText(
                annotated_frame,
                coord_text,
                (center_x - 40, center_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color_bgr,
                1,
            )

        return annotated_frame

    def update_statistics(self, detections):
        """Update detection statistics"""
        for det in detections:
            class_name = det["class"]
            center_x, center_y = det["center"]
            confidence = det["confidence"]

            self.total_detections.set(self.total_detections.get() + 1)
            self.class_counts[class_name].set(self.class_counts[class_name].get() + 1)

            # Add to recent detections
            timestamp = datetime.now().strftime("%H:%M:%S")
            recent_text = f"{timestamp} - {class_name} ({confidence:.2f})"
            self.recent_listbox.insert(0, recent_text)

            # Add to recent coordinates
            coord_text = f"{timestamp} - {class_name}: ({center_x},{center_y})"
            self.coords_listbox.insert(0, coord_text)

            # Keep only last 20 entries for both lists
            if self.recent_listbox.size() > 20:
                self.recent_listbox.delete(tk.END)

            if self.coords_listbox.size() > 20:
                self.coords_listbox.delete(tk.END)

    def capture_frames(self):
        """Capture frames from camera"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect objects
            detections = self.detect_objects(frame)

            # Update statistics
            if detections:
                self.update_statistics(detections)

            # Draw detections
            annotated_frame = self.draw_detections(frame, detections)

            # Put frame in queue
            try:
                self.frame_queue.put_nowait(annotated_frame)
            except queue.Full:
                pass

    def update_display(self):
        """Update video display"""
        if self.running:
            try:
                frame = self.frame_queue.get_nowait()

                # Convert frame to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(frame_pil)

                # Update canvas
                self.video_canvas.delete("all")
                self.video_canvas.create_image(320, 320, image=photo)
                self.video_canvas.image = photo  # Keep reference

                # Update FPS
                self.fps_counter += 1
                if self.fps_counter % 30 == 0:
                    fps_end = datetime.now()
                    fps = 30 / (fps_end - self.fps_start).total_seconds()
                    self.current_fps.set(f"{fps:.1f}")
                    self.fps_start = fps_end

            except queue.Empty:
                pass

        # Schedule next update
        self.root.after(10, self.update_display)

    def start_detection(self):
        """Start real-time detection"""
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return

        try:
            self.cap = cv2.VideoCapture(self.camera_id.get())
            if not self.cap.isOpened():
                messagebox.showerror(
                    "Error", f"Cannot open camera {self.camera_id.get()}"
                )
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

            self.running = True
            self.status_label.config(text="Running", foreground="blue")

            # Update button states
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.capture_btn.config(state=tk.NORMAL)

            # Start capture thread
            self.capture_thread = threading.Thread(target=self.capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()

            # Start display updates
            self.update_display()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection:\n{str(e)}")

    def stop_detection(self):
        """Stop real-time detection"""
        self.running = False

        if self.cap:
            self.cap.release()

        self.status_label.config(text="Stopped", foreground="red")

        # Update button states
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)

        # Clear video canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            320, 320, text="Video Stopped", fill="white", font=("Arial", 16)
        )

    def capture_frame(self):
        """Capture current frame"""
        if not self.running:
            return

        try:
            frame = self.frame_queue.get_nowait()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"realtime_captures/capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            messagebox.showinfo("Success", f"Frame captured!\n{filename}")

        except queue.Empty:
            messagebox.showwarning("Warning", "No frame available to capture")

    def reset_statistics(self):
        """Reset all statistics"""
        self.total_detections.set(0)
        for class_name in self.class_counts:
            self.class_counts[class_name].set(0)

        self.recent_listbox.delete(0, tk.END)
        self.coords_listbox.delete(0, tk.END)
        self.fps_counter = 0
        self.fps_start = datetime.now()

    def on_closing(self):
        """Handle window closing"""
        if self.running:
            self.stop_detection()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = RealTimeTrashDetectorTkinter(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    main()
