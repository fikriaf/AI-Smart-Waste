import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import Counter
import serial


class SampahDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Klasifikasi Sampah - Realtime & Test")

        self.model = None
        self.cap = None
        self.detecting = False

        self.model_path = tk.StringVar(value="No model loaded")
        self.conf_input = tk.DoubleVar(value=0.4)
        self.iou_input = tk.DoubleVar(value=0.5)
        self.conf_used = 0.4
        self.iou_used = 0.5
        self.result_var = tk.StringVar(value="Tidak ada prediksi")
        self.test_image_path = None
        self.selected_camera = tk.IntVar(value=0)

        self.arduino = serial.Serial("COM3", 9600, timeout=1)

        self.setup_ui()
        self.update_frame()

    def setup_ui(self):
        self.tab_control = ttk.Notebook(self.root)
        self.tab_realtime = ttk.Frame(self.tab_control)
        self.tab_testimg = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_realtime, text="Realtime Detection")
        self.tab_control.add(self.tab_testimg, text="Test Image")
        self.tab_control.pack(expand=1, fill="both")

        self.setup_realtime_ui(self.tab_realtime)
        self.setup_testimg_ui(self.tab_testimg)

    def setup_realtime_ui(self, frame):
        control_frame = ttk.Frame(frame)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        ttk.Label(control_frame, text="Kontrol Model", font=("Arial", 12, "bold")).pack(
            pady=5
        )
        ttk.Button(control_frame, text="Load best.pt", command=self.load_model).pack(
            pady=5
        )
        ttk.Label(control_frame, textvariable=self.model_path, wraplength=150).pack(
            pady=10
        )

        self.setup_camera_dropdown(control_frame)

        ttk.Label(control_frame, text="Confidence Threshold:").pack(pady=(15, 0))
        ttk.Entry(control_frame, textvariable=self.conf_input, width=10).pack()

        ttk.Label(control_frame, text="IoU Threshold:").pack(pady=(10, 0))
        ttk.Entry(control_frame, textvariable=self.iou_input, width=10).pack()

        ttk.Button(
            control_frame, text="Simpan Konfigurasi", command=self.apply_config
        ).pack(pady=5)
        self.toggle_btn = ttk.Button(
            control_frame, text="Deteksi: OFF", command=self.toggle_detection
        )
        self.toggle_btn.pack(pady=5)

        self.video_label = ttk.Label(frame)
        self.video_label.grid(row=0, column=1, padx=10, pady=10)

        output_frame = ttk.Frame(frame)
        output_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")
        ttk.Label(output_frame, text="Hasil Deteksi", font=("Arial", 12, "bold")).pack(
            pady=5
        )
        ttk.Label(output_frame, textvariable=self.result_var, wraplength=150).pack(
            pady=10
        )

    def setup_camera_dropdown(self, parent):
        ttk.Label(parent, text="Pilih Kamera:").pack(pady=(10, 0))

        self.available_camera_indexes = []
        for i in range(5):
            print(f"🔍 Mencoba kamera index {i}")
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.read()[0]:
                print(f"✅ Kamera index {i} tersedia")
                self.available_camera_indexes.append(i)
            else:
                print(f"❌ Kamera index {i} gagal")
            cap.release()

        if 1 in self.available_camera_indexes:
            self.selected_camera.set(1)
        elif self.available_camera_indexes:
            self.selected_camera.set(self.available_camera_indexes[0])
        else:
            self.selected_camera.set(-1)

        self.cam_menu = ttk.OptionMenu(
            parent,
            self.selected_camera,
            self.selected_camera.get(),
            *self.available_camera_indexes,
            command=self.change_camera,
        )
        self.cam_menu.pack()

        self.change_camera()

    def change_camera(self, *args):
        cam_index = self.selected_camera.get()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"✅ Kamera index {cam_index} dibuka")

    def setup_testimg_ui(self, frame):
        control_frame = ttk.Frame(frame)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        ttk.Label(control_frame, text="Kontrol Model", font=("Arial", 12, "bold")).pack(
            pady=5
        )
        ttk.Button(control_frame, text="Load best.pt", command=self.load_model).pack(
            pady=5
        )
        ttk.Label(control_frame, textvariable=self.model_path, wraplength=150).pack(
            pady=10
        )
        ttk.Button(control_frame, text="Input Gambar", command=self.select_image).pack(
            pady=5
        )

        ttk.Label(control_frame, text="Confidence Threshold:").pack(pady=(15, 0))
        ttk.Entry(control_frame, textvariable=self.conf_input, width=10).pack()

        ttk.Label(control_frame, text="IoU Threshold:").pack(pady=(10, 0))
        ttk.Entry(control_frame, textvariable=self.iou_input, width=10).pack()

        ttk.Button(
            control_frame, text="Simpan Konfigurasi", command=self.apply_config
        ).pack(pady=5)
        ttk.Button(control_frame, text="Start Test", command=self.run_test_image).pack(
            pady=10
        )

        self.img_preview_label = ttk.Label(frame)
        self.img_preview_label.grid(row=0, column=1, padx=10, pady=10)

        output_frame = ttk.Frame(frame)
        output_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")
        ttk.Label(output_frame, text="Hasil Deteksi", font=("Arial", 12, "bold")).pack(
            pady=5
        )
        self.result_text = tk.Text(output_frame, height=10, width=30, wrap="word")
        self.result_text.pack(pady=10)
        self.result_text.configure(state="disabled")

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model PyTorch", "*.pt")])
        if file_path:
            self.model = YOLO(file_path)
            self.model_path.set(f"Model: {file_path.split('/')[-1]}")

    def toggle_detection(self):
        self.detecting = not self.detecting
        self.toggle_btn.config(text=f"Deteksi: {'ON' if self.detecting else 'OFF'}")

    def apply_config(self):
        try:
            self.conf_used = float(self.conf_input.get())
            self.iou_used = float(self.iou_input.get())
        except ValueError:
            print("Nilai konfigurasi tidak valid")

    def update_frame(self):
        if self.cap and self.cap.isOpened() and self.tab_control.index("current") == 0:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                self.result_var.set("⚠️ Kamera aktif, tapi tidak kirim gambar.")
                self.root.after(500, self.update_frame)
                return

            display_frame = frame.copy()

            if self.model and self.detecting:
                results = self.model.predict(
                    display_frame, conf=self.conf_used, iou=self.iou_used, verbose=False
                )
                annotated = results[0].plot()

                names = self.model.names
                classes = results[0].boxes.cls if results[0].boxes is not None else []
                labels = (
                    [names[int(cls)] for cls in classes] if len(classes) > 0 else []
                )

                if labels:
                    self.arduino.write(b"ON\n")
                else:
                    self.arduino.write(b"OFF\n")

                self.result_var.set(", ".join(labels) if labels else "Tidak terdeteksi")
            else:
                annotated = display_frame
                self.result_var.set("Deteksi dimatikan")

            img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return
        image = cv2.imread(file_path)
        if image is None:
            self.result_var.set("⚠️ Gagal membuka gambar.")
            return

        self.test_image_path = file_path
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.img_preview_label.imgtk = imgtk
        self.img_preview_label.configure(image=imgtk)

    def update_result_text(self, text):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state="disabled")

    def run_test_image(self):
        if self.model is None:
            self.result_var.set("⚠️ Load model terlebih dahulu!")
            return
        if not self.test_image_path:
            self.result_var.set("⚠️ Pilih gambar terlebih dahulu!")
            return
        image = cv2.imread(self.test_image_path)
        if image is None:
            self.result_var.set("⚠️ Gagal membaca gambar.")
            return

        results = self.model.predict(
            image, conf=self.conf_used, iou=self.iou_used, verbose=False
        )
        annotated = results[0].plot()

        names = self.model.names
        classes = results[0].boxes.cls if results[0].boxes is not None else []
        if len(classes) > 0:
            class_ids = [int(cls) for cls in classes]
            class_counts = Counter(class_ids)
            labels_with_count = [
                f"{names[class_id]} ({class_id}): {count}x"
                for class_id, count in class_counts.items()
            ]
            output = "\n".join(labels_with_count)
            self.update_result_text(output)
        else:
            self.update_result_text("Tidak terdeteksi")

        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.img_preview_label.imgtk = imgtk
        self.img_preview_label.configure(image=imgtk)

    def exit_app(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SampahDetectionApp(root)
    root.mainloop()
