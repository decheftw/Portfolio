from sys import *
import signal
from mrt import *
import os



def main():
    sock = socket(AF_INET, SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    connect = ("127.0.0.1", 5005)
    window = mrt_connect(sock, connect)
    data = sys.stdin.read()
    file = open(str(sock.getsockname()), "a+")
    file.write(data)
    file.close()
    mrt_send(sock, window, str(sock.getsockname()))
    os.remove(str(sock.getsockname()))
    mrt_disconnect(sock)


main()
