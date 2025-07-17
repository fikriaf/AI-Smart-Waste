import socket
import tkinter as tk
from tkinter import filedialog, simpledialog
import threading
import os
import struct

PORT = 5001
BUFFER_SIZE = 1024

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Tidak diketahui"

def start_ui():
    last_ip = [None]

    def start_server():
        def run_server():
            local_ip = get_local_ip()
            status_label.config(text=f"IP Lokal Anda: {local_ip}\nMenunggu koneksi di port {PORT}...")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.bind(('', PORT))
                server.listen()
                while True:
                    conn, addr = server.accept()
                    threading.Thread(target=handle_client, args=(conn, addr)).start()

        def handle_client(conn, addr):
            with conn:
                log_output(f"Terhubung dengan {addr}. Menerima metadata...")
                try:
                    header = conn.recv(1).decode()  # 'T' atau 'F'
                    if header == 'F':  # File
                        filename_len_data = conn.recv(4)
                        if not filename_len_data:
                            log_output("Gagal menerima panjang nama file.")
                            return
                        filename_len = struct.unpack('!I', filename_len_data)[0]
                        filename = conn.recv(filename_len).decode()
                        save_path = os.path.join(os.path.expanduser("~/Downloads"), filename)
                        log_output(f"Menerima file: {filename}...")

                        with open(save_path, 'wb') as f:
                            while True:
                                data = conn.recv(BUFFER_SIZE)
                                if not data:
                                    break
                                f.write(data)
                        log_output(f"✅ File berhasil disimpan: {save_path}\n")

                    elif header == 'T':  # Text
                        text_len_data = conn.recv(4)
                        if not text_len_data:
                            log_output("Gagal menerima panjang teks.")
                            return
                        text_len = struct.unpack('!I', text_len_data)[0]
                        message = conn.recv(text_len).decode()
                        log_output(f"📩 Pesan diterima: {message}")

                    else:
                        log_output("❌ Format data tidak dikenal.")

                except Exception as e:
                    log_output(f"❌ Terjadi kesalahan: {e}")

        threading.Thread(target=run_server, daemon=True).start()

    def send_file():
        ip = get_target_ip()
        if not ip:
            return

        file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if not file_path:
            return

        filename = os.path.basename(file_path)
        filename_bytes = filename.encode()
        filename_len = struct.pack('!I', len(filename_bytes))

        def run_client():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                    client.connect((ip, PORT))
                    log_output(f"Mengirim file {filename} ke {ip}...")
                    client.sendall(b'F')  # Header file
                    client.sendall(filename_len)
                    client.sendall(filename_bytes)
                    with open(file_path, 'rb') as f:
                        while (data := f.read(BUFFER_SIZE)):
                            client.sendall(data)
                    status_label.config(text="✅ File berhasil dikirim.")
                    log_output(f"✅ File {filename} berhasil dikirim.")
                    again_button.pack(pady=10)
            except Exception as e:
                status_label.config(text=f"❌ Gagal mengirim file: {e}")

        threading.Thread(target=run_client).start()

    def send_text():
        ip = get_target_ip()
        if not ip:
            return

        message = simpledialog.askstring("Kirim Teks", "Masukkan pesan yang akan dikirim:")
        if not message:
            return

        message_bytes = message.encode()
        message_len = struct.pack('!I', len(message_bytes))

        def run_client():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                    client.connect((ip, PORT))
                    log_output(f"Mengirim pesan ke {ip}...")
                    client.sendall(b'T')  # Header teks
                    client.sendall(message_len)
                    client.sendall(message_bytes)
                    status_label.config(text="✅ Pesan berhasil dikirim.")
                    log_output(f"✅ Pesan berhasil dikirim.")
                    again_button.pack(pady=10)
            except Exception as e:
                status_label.config(text=f"❌ Gagal mengirim pesan: {e}")

        threading.Thread(target=run_client).start()

    def get_target_ip():
        if last_ip[0] is None:
            ip = simpledialog.askstring("Alamat IP Tujuan", "Masukkan IP penerima:")
            if not ip:
                return None
            last_ip[0] = ip
        return last_ip[0]

    def pilih_mode(mode):
        send_file_button.pack_forget()
        send_text_button.pack_forget()
        recv_button.pack_forget()
        if mode == "terima":
            start_server()

    def log_output(text):
        output_text.config(state="normal")
        output_text.insert(tk.END, text + "\n")
        output_text.config(state="disabled")
        output_text.see(tk.END)

    # UI Setup
    root = tk.Tk()
    root.title("Kirim/Terima File atau Teks via WiFi")
    root.geometry("520x450")

    title_label = tk.Label(root, text="Pilih Mode", font=("Helvetica", 16))
    title_label.pack(pady=10)

    send_file_button = tk.Button(root, text="KIRIM FILE", width=25, command=send_file)
    send_text_button = tk.Button(root, text="KIRIM TEKS", width=25, command=send_text)
    recv_button = tk.Button(root, text="TERIMA FILE / TEKS", width=25, command=lambda: pilih_mode("terima"))
    send_file_button.pack(pady=5)
    send_text_button.pack(pady=5)
    recv_button.pack(pady=10)

    status_label = tk.Label(root, text="", wraplength=460, justify="center", fg="blue")
    status_label.pack(pady=5)

    output_text = tk.Text(root, height=12, width=60, state="disabled", wrap="word")
    output_text.pack(pady=10)

    again_button = tk.Button(root, text="Kirim Lagi", width=25, command=lambda: [send_file_button.pack(pady=5), send_text_button.pack(pady=5), recv_button.pack(pady=10), again_button.pack_forget()])
    again_button.pack_forget()

    root.mainloop()

if __name__ == "__main__":
    start_ui()
