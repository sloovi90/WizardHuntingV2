import socket
import sys

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("localhost", 5656))
print(s.recv(10).decode());
print(s.recv(10).decode());