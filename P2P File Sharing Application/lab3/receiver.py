from mrt import Server, ServerConnection
import time
import sys

'''
Shikhar Sinha, Receiver Program
For MRT Lab3
'''

if __name__ == '__main__':
    server = Server.mrt_open(12345)
    # n = int(sys.argv[1])
    # connections = []
    # # time.sleep(15)
    # # connections.extend(server.mrt_accept_all())
    # for i in range(n):
    #     connections.append(server.mrt_accept1())
    # # print("\naccepted n connections \n")
    # for i in range(n):
    #     connection = connections[i]
    #     while connection.connected == True:
    #         data = connection.mrt_receive1()
    #         if data != None:
    #             print(bytearray(data).decode("ascii"))
    #     #     if ServerConnection.mrt_probe([connection]) != None:
    #     #         data = connection.mrt_receive1()
    #     #         print(bytearray(data).decode("ascii"))
    #     # if ServerConnection.mrt_probe([connection]) != None:
    #     #     print(bytearray(data).decode("ascii"))
    #     # print("done with connection " + str(i))
    
    file1 = open('../../mrt_test_write.txt', 'a')
    connection = server.mrt_accept1()
    while True and connection.connected == True:
        file1.write(connection.mrt_receive1().decode("ascii"))
    file1.close()

    server.mrt_close()
    # print("server is closed")
