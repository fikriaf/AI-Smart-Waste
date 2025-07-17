import serial
import time

arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)

# Simulasi hasil dari proses AI atau apapun
result = "plastic"

if result == "plastic":
    arduino.write(b'ON\n')
else:
    arduino.write(b'OFF\n')
