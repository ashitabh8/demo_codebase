import socket

HOST = "172.20.10.12"
PORT = 8080

data = bytes([1, 2, 3])

s = socket.socket()
s.connect((HOST, PORT))
print("Connected")

s.sendall(data)
print("Sent:", list(data))

reply = s.recv(3)
print("Got back raw:", reply)
print("Got back list:", list(reply))

s.close()