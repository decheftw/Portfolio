import sys
from node import Node, NeighborNode
import time
import json
from mrt import Server, ReceiverConnection, SenderConnection

if __name__ == '__main__':
    own_port = int(sys.argv[1])
    port_1 = int(sys.argv[3])
    # port_2 = int(sys.argv[4])
    print(own_port)
    print(port_1)
    new_node = Node.start_node(own_port, sys.argv[2])
    new_node.connect('71.184.117.14', port_1, '192.168.1.175', port_1)

    # new_node.connect('127.0.0.1', port_1)
    # new_node.neighbors[('127.0.0.1', port_1)].control_sender_connection.mrt_send(bytes(json.dumps(["test", {"bs": "yeah"}, "EOF"]), encoding="ascii"))
    # time.sleep
    new_node.add_file("small_pride.txt")

    # time.sleep(3)

    # time.sleep(3)
    # print(new_node.file_directory)
    # exit()
