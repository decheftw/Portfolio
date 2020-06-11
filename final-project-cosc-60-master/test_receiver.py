import sys
from node import Node, NeighborNode
import time
import json
from mrt import Server, ReceiverConnection, SenderConnection

if __name__ == '__main__':
    own_port = int(sys.argv[1])
    # port_1 = int(sys.argv[3])
    print(own_port)
    new_node = Node.start_node(own_port, sys.argv[2])
    time.sleep(10)
    # print(new_node.file_directory)
    print("requesting file")
    new_node.request("small_pride.txt")
    # print("finished downloading file")
 