import json
from mrt import Server, ReceiverConnection, SenderConnection
import os
import sys
import tempfile
import errno
import threading
import time
import socket
import copy
import base64


import hashlib
import stun
import stun_client

BROADCAST = 'broadcast'

DEFAULT_DIR = "./p2p_dir"
MSG_TYPE = 0
MSG_DATA = 1
MSG_EOF = 2


class Node:
    def __init__(self, sock, server, storage, external_ip, external_port, private_ip, private_port):

        self.sock = sock
        self.server = server  # server object
        self.storage = storage  # directory path
        self.neighbors = dict()  # map ip/port to NeighborNode Object

        self.server_lock = threading.Lock()
        self.broadcast_lock = threading.Lock()

        # nat_type, external_ip, external_port = stun.get_ip_info()
        self.external_ip = external_ip
        self.external_port = external_port
        self.private_ip = private_ip
        self.private_port = private_port
        print("external: " + self.external_ip + ":" + str(self.external_port))
        print("private: " + self.private_ip + ":" + str(self.private_port))



        self.file_directory = dict()  # map filename -> (min hops, neighbor providing min hops ip, port)

        self.username = None # See peer_client.py for how this is set
        # maps ip/port -> last received chat if neighbor
        # ip/port -> our current chat # for self
        self.chat_numbers = {(self.external_ip, self.external_port): 0}

        self.node_broadcast_numbers = dict()  # keep track of the latest broadcast number associated with a given node
        self.node_broadcast_numbers[(self.external_ip, self.external_port)] = 0
        

        self.forward = {}  # A list of files to be forwarded identifier: (filename, fileno) to data: neighbor

    @staticmethod
    def start_node(port, directory_path, private=True):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('', port))
        except OSError:
            print("Error with socket!!")
            return None

        external_ip, external_port = stun_client.client(sock)
        if external_ip == None:
            print("Error! With STUN Please try again")
            return None


        private_ip = None
        try:
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_sock.connect(("8.8.8.8", 80))
            # print("localaddress: " + temp_sock.getsockname()[0])
            private_ip = temp_sock.getsockname()[0]
            temp_sock.close()
        except OSError:
            print("Error in startup process!")
            return None

        if private:
            external_ip = private_ip
            external_port = port

        server = Server.mrt_open_sock(sock, port)
        if server == None:
            print("Error: Unable to start Server")
            return None
        storage = directory_path
        try:
            testfile = tempfile.TemporaryFile(dir=storage)
            testfile.close()
        except (OSError, IOError):
            print("unable to use supplied directory")
            try:
                os.mkdir(DEFAULT_DIR)
            except (OSError, IOError) as e:
                if e.errno == errno.EACCES or e.errno == errno.EEXIST:  # 13, 17
                    print("unable to create default directory")
                return None
            storage = DEFAULT_DIR
        ret = Node(sock, server, storage, external_ip, external_port, private_ip, port)
        ret.external_ip = external_ip
        ret.external_port = external_port
        ret.storage = storage

        server_thread = threading.Thread(target=ret.server_handler, args=())
        server_thread.daemon = True
        server_thread.start()

        return ret

    def server_handler(self):
        while self.server.open:
            time.sleep(1)
            new_connections = self.server.mrt_accept_all()
            for new_connection in new_connections:
                # print("Received Connection at Address: " + str(new_connection.src_address))

                recvd_msg = NeighborNode.receive_message_2(new_connection, '')
                parsed_msg = (recvd_msg[0])
                if parsed_msg[MSG_TYPE] != "addr_info":
                    print("BIG ERROR, dropping this connection")
                    continue
                msg_ip = parsed_msg[MSG_DATA]['external_ip']
                msg_port = int(parsed_msg[MSG_DATA]['external_port'])
                msg_private_ip = parsed_msg[MSG_DATA]['private_ip']
                msg_private_port = int(parsed_msg[MSG_DATA]['private_port'])
                target_ip = msg_ip
                target_port = msg_port
                with self.server_lock:
                    if self.neighbors.get((msg_ip, msg_port)) != None:
                        if self.neighbors[(msg_ip, msg_port)].ready == False:
                            # print("completing the connection")
                            self.neighbors[
                                (msg_ip, msg_port)].control_receiver_connection = new_connection
                            self.neighbors[(msg_ip, msg_port)].ready = True
                        else:
                            print("This connection is already ready to go! Nice try")
                            continue
                    else:
                        # print("we haven't seen this ip/port before! ")
                        new_neighbor = NeighborNode(self)
                        new_neighbor.control_receiver_connection = new_connection
                        target_ip = msg_ip
                        target_port = msg_port
                        new_sender_connection = self.server.mrt_connect(new_connection.src_address[0], new_connection.src_address[1])
                        # print("src address: " + str(new_connection.src_address))
                        addr_info_msg = json.dumps(["addr_info", {'external_ip': str(self.external_ip),
                                                                  'external_port': str(self.external_port),
                                                                'private_ip': str(self.private_ip),
                                                                'private_port': str(self.private_port)}])
                        new_neighbor.control_sender_connection = new_sender_connection

                        new_neighbor.send_neighbor_msg_2(addr_info_msg)

                        new_neighbor.ready = True
                        self.neighbors[(msg_ip, msg_port)] = new_neighbor
                    self.neighbors[(msg_ip, msg_port)].neighbor_address = (
                    msg_ip, msg_port)
                    neighbor_thread = threading.Thread(
                        target=self.neighbors[(msg_ip, msg_port)].neighbor_node_handler,
                        args=(recvd_msg[1],))
                    neighbor_thread.daemon = True
                    neighbor_thread.start()

                    for filename in self.file_directory:
                        # hops = hops + 1 of all files in there
                        hops, ip, port = self.file_directory[filename]
                        msg_data = self.create_file_message(filename, hops)
                        self.broadcast(msg_data)

                    

    def broadcast(self, msg_data):
        # print("broadcast called with: " + str(msg_data))
        msg_data['number_of_hops'] += 1

        for neighbor in self.neighbors.values():
            # msg = ["broadcast", msg_data, "EOF"]
            #, recvd_msg neighbor.send_neighbor_msg(json.dumps(msg))
            msg = ["broadcast", msg_data]
            neighbor.send_neighbor_msg_2(json.dumps(msg))

    # Create a chat message 
    def create_chat_msg(self, text):
        our_addr = (self.external_ip, self.external_port)
        msg_data = {
            'username': self.username,
            'text': text,
            'num': self.chat_numbers[our_addr],
            'origin_ip': self.external_ip,
            'origin_port': self.external_port,
        }
        self.chat_numbers[our_addr] += 1

        self.send_chat(msg_data)

    # Send a chat msg created by the user OR forward one from another user
    def send_chat(self, msg_data):
        for neighbor in self.neighbors.values():
            msg = ["chat", msg_data]
            neighbor.send_neighbor_msg_2(json.dumps(msg))

    def node_close(self):
        self.server.mrt_close()


    def add_file(self, filename):
        full_path_name = self.storage + "/" + filename
        if not os.path.exists(full_path_name):
            print("oh no")
            return ("The file you have named does not exist.")

        msg_data = self.create_file_message(filename)

        # add to file_directory
        self.file_directory[filename] = (0, self.external_ip, self.external_port)
        with self.server_lock:
            self.broadcast(msg_data)

    def create_file_message(self, filename, number_of_hops=0):
        msg_data = dict()
        msg_data['filename'] = filename
        msg_data['neighbor_ip'] = self.external_ip
        msg_data['neighbor_port'] = self.external_port
        msg_data['number_of_hops'] = number_of_hops

        return msg_data



    def connect(self, ip_addr, port, private_ip=None, private_port=None):
        new_neighbor = None
        with self.server_lock:
            if self.neighbors.get((ip_addr, port)) != None:
                print("Already connected to this host!")
                new_neighbor = self.neighbors[(ip_addr, port)]
            else:
                new_neighbor = NeighborNode(self)
                target_ip = ip_addr
                target_port = port
                if private_ip != None and private_port != None and ip_addr == self.external_ip:
                    print("using private ip")
                    target_ip = private_ip
                    target_port = private_port

                new_sender_connection = self.server.mrt_connect(target_ip, target_port)
                self.neighbors[(ip_addr, port)] = new_neighbor
                new_neighbor.control_sender_connection = new_sender_connection

                addr_info_msg = json.dumps(
                    ["addr_info", {'external_ip': str(self.external_ip),
                                    'external_port': str(self.external_port),
                                    'private_ip': str(self.private_ip),
                                    'private_port': str(self.private_port)}])
                new_neighbor.send_neighbor_msg_2(addr_info_msg)

        while new_neighbor.ready == False:
            # print("waiting for corresponding connection")
            time.sleep(2)

    def request(self, filename):
        address = self.file_directory.get(filename)
        if address == None:
            print("file not found")
            return
        else:
            request_message = json.dumps(
                ["request", {'filename': filename, 'neighborip': self.external_ip, 'neighborport': self.external_port}])
            # print(address)
            origin = self.neighbors.get((address[1], address[2]))
            origin.send_neighbor_msg_2(request_message)

    def _encode(self, data):
        return base64.encodebytes(data).decode('ascii')

    def _decode(self, data):
        return base64.decodebytes(data.encode('ascii'))

    def send_file(self, filename, neighbor):
        file = open(self.storage + "/" + filename, "rb")
        data = self._encode(file.read(10000))

        i = 1
        message = None
        while data != "":
            data2 = self._encode(file.read(10000))
            if data2 == "":
                message = json.dumps(["dataf", {"filename": filename, "data": copy.deepcopy(data),
                                                "fileno": str(i)}])  # if it's the final packet
            else:
                message = json.dumps(["data", {"filename": filename, "data": copy.deepcopy(data), "fileno": str(i)}])
            neighbor.send_neighbor_msg_2(message)
            data = data2
            i += 1
        # print(i)
        # print("Finished sending file")
        file.close()
        # file2.close()


    def receive_request(self, filename, neighbor):
        address = self.file_directory.get(filename)
        if address[1] == self.external_ip and address[2] == self.external_port:
            self.send_file(filename, neighbor)
        else:
            if (filename, 1) not in self.forward:
                self.forward[(filename, 1)] = []
            self.forward[(filename, 1)].append(neighbor)
            self.request(filename)



class NeighborNode:
    def __init__(self, node):
        self.control_receiver_connection = None
        self.control_sender_connection = None  # how/when to add to these things
        self.neighbor_address = None
        self.neighbor_lock = threading.Lock()
        self.ready = False  # maybe keep in this state until bidirectional communications are established
        self.node = node
        self.storage = ""
        self.i = 0



    @staticmethod
    def receive_message_2(receiver_connection, partial_msg):
        eom = False
        msg = partial_msg
        msg_to_handle = ''
        msg_len = 0

        split_res = msg.split('[', maxsplit=1)

        if len(split_res) > 1:

            msg_len = int(split_res[0])
            split_res[1] = '[' + split_res[1]

            if len(split_res[1]) >= msg_len:

                msg_to_handle = split_res[1][:msg_len]
                msg = split_res[1][msg_len:]
                eom = True

            else:

                msg_to_handle = split_res[1]

        while eom == False:
            if ReceiverConnection.mrt_probe([receiver_connection]) != None:
                to_add = receiver_connection.mrt_receive1().decode("ascii")

                if msg_len == 0:
                    msg += to_add
                    split_res = msg.split('[', maxsplit=1)
                    if len(split_res) > 1:
                        # print("thing to be split: " + msg)
                        msg_len = int(split_res[0])
                        # print("msg length: " + str(msg_len))
                        msg_to_handle = '[' + split_res[1]
                else:
                    msg_to_handle += to_add
            if msg_len > 0:
                if len(msg_to_handle) >= msg_len:
                    msg = msg_to_handle[msg_len:]
                    msg_to_handle = msg_to_handle[:msg_len]
                    eom = True

        return (json.loads(msg_to_handle, strict=True), msg)

    def neighbor_node_handler(self,
        partial_msg):  # a thread func started likely in the server handler incl. upon receipt of response_to_connect
        # Handle data from the connections
        # print("starting the neighbor_node for" + str(self.neighbor_address))
        msg = partial_msg
        while self.control_receiver_connection.connected == True and self.control_sender_connection.connected == True:

            # read in a message
            # recvd_msg = NeighborNode.receive_message(self.control_receiver_connection,msg)
            recvd_msg = NeighborNode.receive_message_2(self.control_receiver_connection, msg)

            parsed_msg = recvd_msg[0]
            # logfile.write(parsed_msg[MSG_DATA])

            msg = recvd_msg[1]
            if parsed_msg[MSG_TYPE] == 'broadcast':
                self.handle_broadcast_msg(parsed_msg)
            elif parsed_msg[MSG_TYPE] == 'data':
                self.handle_data_message(parsed_msg)
            elif parsed_msg[MSG_TYPE] == 'dataf':
                self.handle_data_message(parsed_msg)
            elif parsed_msg[MSG_TYPE] == 'request':
                self.node.receive_request(parsed_msg[MSG_DATA]['filename'], self.node.neighbors[
                    (parsed_msg[MSG_DATA]['neighborip'], parsed_msg[MSG_DATA]['neighborport'])])
            elif parsed_msg[MSG_TYPE] == 'chat':
                self.handle_chat_msg(parsed_msg)
                # should we just print the message and then forward it like a broadcast msg
            else:
                print("Msg Type unknown or unimplemented")
        print("Shutting down node")




    def send_neighbor_msg_2(self, msg_str):
        msg_len = len(msg_str)
        new_msg_str = str(msg_len) + msg_str
        with self.neighbor_lock:
            self.control_sender_connection.mrt_send(bytearray(new_msg_str, encoding="ascii"))


    def handle_broadcast_msg(self, parsed_msg):
        msg_data = parsed_msg[MSG_DATA]

        filename = msg_data['filename']
        if filename in self.node.file_directory:
            prev_hops, _, _ = self.node.file_directory[filename]
            # Ignore this broadcast if we've seen a better way to reach the file
            if msg_data['number_of_hops'] >= prev_hops:
                return

        # Otherwise save this path and pass the message along
        self.node.file_directory[filename] = (msg_data['number_of_hops'],
                                              msg_data['neighbor_ip'],
                                              msg_data['neighbor_port'])

        # Update us as the neighbor
        msg_data['neighbor_ip'] = self.node.external_ip
        msg_data['neighbor_port'] = self.node.external_port
        with self.node.server_lock:
            self.node.broadcast(msg_data)


    def handle_chat_msg(self, parsed_msg):
        msg_data = parsed_msg[MSG_DATA]
        
        username = msg_data['username']
        text = msg_data['text']
        origin_addr = msg_data['origin_ip'], msg_data['origin_port']
        chat_num = msg_data['num']

        # print the chat if
        #  1. this is the first time seeing this neighbor chatting us
        #  2. the chat from this node has a higher counter value
        if (origin_addr not in self.node.chat_numbers
                or chat_num > self.node.chat_numbers[origin_addr]):
            self.node.chat_numbers[origin_addr] = chat_num

            print(username + ": " + text)

            with self.node.server_lock:
                self.node.send_chat(msg_data)
                
    
    def handle_data_message(self, parsed_msg):
        filename = parsed_msg[MSG_DATA]['filename']
        fileno = int(parsed_msg[MSG_DATA]['fileno'])

        # Our job is to forward the file
        if (filename, fileno) in self.node.forward:
            destinations = self.node.forward[(filename, fileno)]
            
            destination = destinations.pop()
            destination.send_neighbor_msg_2(json.dumps(parsed_msg))
            
            if (filename, fileno + 1) not in self.node.forward:
                self.node.forward[(filename, fileno + 1)] = []
            
            self.node.forward[(filename, fileno + 1)].append(destination)

            if len(destinations) == 0:
                del self.node.forward[(filename, fileno)]

        # We requested the file, therefore, we're downloading it
        else:
            data = self.node._decode(parsed_msg[MSG_DATA]['data'])
            with open(self.node.storage + '/' + filename, "ab") as f:
                f.write(data)


            # If this is the last packet
            #   1. Tell the user we are done downloading
            #   2. Tell other nodes that we have this file locally
            if parsed_msg[MSG_TYPE] == "dataf":
                print("File " + filename + " is done downloading")
                self.node.add_file(filename)

