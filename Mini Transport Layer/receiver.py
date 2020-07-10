from mrt import *

connection = ("127.0.0.1", 5005)
window_size = 10
n = int(sys.argv[1])


def main():
    server = mrt_open("127.0.0.1", 5005)
    server.start()
    i = 0
    while i < n:
        connection = mrt_accept1(server)
        mrt_receive1(connection, server)
        i += 1
    mrt_close(server)

main()