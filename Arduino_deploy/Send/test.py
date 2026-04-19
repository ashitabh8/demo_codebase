import socket
import numpy as np

HOST = "172.20.10.12"
PORT = 8080

# Random 120x10 array with values 0-255
arr = np.random.randint(0, 256, (120, 10), dtype=np.uint8)
data = arr.flatten().tobytes()  # 1200 bytes

print(f"Array shape: {arr.shape}, total bytes: {len(data)}")
print(f"First row: {arr[0]}")

s = socket.socket()
s.connect((HOST, PORT))
print("Connected")
s.sendall(data)
print(f"Sent {len(data)} bytes")

# Wait for Arduino's confirmation (sending back total bytes received as 2 bytes)
reply = s.recv(2)
received_count = int.from_bytes(reply, 'big')
print(f"Arduino confirmed receiving: {received_count} bytes")

s.close()