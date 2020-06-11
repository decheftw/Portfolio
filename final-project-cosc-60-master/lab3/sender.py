import sys
import socket
import time
from mrt import SenderConnection
'''
Shikhar Sinha
Sender Program to send data from stdin to the connected server
'''

if __name__ == '__main__':
    # data_to_send = sys.stdin.read()
    # port = int(sys.argv[1])
    connection = SenderConnection.mrt_connect('localhost', 12345)
    print("Hooray! connected")
    print(connection.server_window_size)
    file1 = open('../test_dir_1/small_pride.txt', 'rb')
    data_to_send = str(file1.read(1000000))
    connection.mrt_send(bytes(data_to_send,encoding="ascii"))
    connection.mrt_disconnect()
    print("connection closed")