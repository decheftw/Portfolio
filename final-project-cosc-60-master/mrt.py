import socket
import threading
import time
import zlib
import random
'''
Shikhar Sinha
Professor Joosten
Computer Networks Spring 2020 Lab 3
This file contains code implementing the MRT Layer

'''

# Constants
INIT_PACKET = 512
DEFAULT_PORT = 22222 # default port
MAX_PACKET = 1200 
TIMEOUT = 0.4 # 400 milliseconds
HEADER_LEN = 15 # length of packet header
CHECKSUM_3 = 0
CHECKSUM_2 = 1
CHECKSUM_1 = 2
CHECKSUM_0 = 3
SEQUENCE_NUM_7 = 4
SEQUENCE_NUM_6 = 5
SEQUENCE_NUM_5 = 6
SEQUENCE_NUM_4 = 7
SEQUENCE_NUM_3 = 8
SEQUENCE_NUM_2 = 9
SEQUENCE_NUM_1 = 10
SEQUENCE_NUM_0 = 11
FLAG = 12
LEN_1 = 13
LEN_0 = 14
ACK_NUM_7 = 4
ACK_NUM_6 = 5
ACK_NUM_5 = 6
ACK_NUM_4 = 7
ACK_NUM_3 = 8
ACK_NUM_2 = 9
ACK_NUM_1 = 10
ACK_NUM_0 = 11
WINDOW_SIZE_1 = 13
WINDOW_SIZE_0 = 14
ACK_FLAG = 0x40
SYN_FLAG = 0x80
DIS_FLAG = 0x20
FIN_FLAG = 0x10
CLI_ACK_FLAG = 0x08

# Params to change/fiddle with
MAX_OUTBUFFER = 4096
RECEIVER_WINDOW = 2048
INIT_SENDER_WINDOW = 2048
PACKET_SIZE = 1024

# lock = threading.Lock()



# mrt_open, accept1, accept_all, and close
class Server:

    def __init__(self, sock, port):
        self.sock = sock
        self.port = port
        self.ready_connections = list()
        self.ready_addresses = set()
        self.pending_connections_dict = dict() # key = address
        self.in_connections = dict() # index a connection object by its port,ip combo
        self.open = True
        self.out_connections = dict() # index a connection object by its port, ip combo
        self.data_lock = threading.Lock()
        self.connect_lock = threading.Lock()

    def verify_crc(self, packet):
        packet_crc = (packet[0] << 24) + (packet[1] << 16) + (packet[2] << 8) + (packet[3])
        return zlib.crc32(packet[SEQUENCE_NUM_7:]) == packet_crc

    @staticmethod
    # should we delete the port parameter
    def mrt_open(port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('', port))
        except OSError:
            print("open_error: unable to connect at " + str(port))
            print("\nAttempting to connect at " + str(DEFAULT_PORT))
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind(('', DEFAULT_PORT))
            except OSError:
                print("unable to connect, please try again with a different port")
                return None
            # print("opening server at port" + str(DEFAULT_PORT))
            server = Server(sock, DEFAULT_PORT)
            packet_thread = threading.Thread(target=server.packet_handler, args=())
            packet_thread.start()
            # print("server started at default port")
            return server
        sock.settimeout(2)
        server = Server(sock, port)
        packet_thread = threading.Thread(target=server.packet_handler, args=())
        packet_thread.start()
        print("server started at input port")
        return server

    @staticmethod
    def mrt_open_sock(sock, port):
        sock.settimeout(2)
        server = Server(sock, port)
        packet_thread = threading.Thread(target=server.packet_handler, args=())
        packet_thread.start()
        return server


    def mrt_accept1(self):
        connection = None
        while connection == None:
            try:
                connection = self.ready_connections.pop(0)
            except IndexError:
                connection = None
        with self.connect_lock:
            self.ready_addresses.remove(connection.src_address)
            self.pending_connections_dict.pop(connection.src_address)
            connection.connected = True
            self.in_connections[connection.src_address] = connection
            # print("Connected! Sequence Number is " + str(connection.sequence_number) + ", src address is " + str(connection.src_address))
        return connection
    
    def mrt_accept_all(self):
        new_connections = []
        with self.connect_lock:
            num_available_connections = len(self.ready_connections)
            i = 0
            while i < num_available_connections:
                connection = self.ready_connections.pop(0)
                self.ready_addresses.remove(connection.src_address)
                self.pending_connections_dict.pop(connection.src_address)
                new_connections.append(connection)
                self.in_connections[connection.src_address] = connection
                connection.connected = True
                i += 1
        return new_connections
    
    def mrt_close(self):
        for out_connection in self.out_connections.values():
            out_connection.mrt_disconnect()
        for connection in self.in_connections.values():
            try:
                with self.data_lock:
                    self.sock.sendto(connection.build_receiver_packet(FIN_FLAG), connection.src_address)
            except OSError:
                continue

        for connection in self.pending_connections_dict.values():
            try:
                with self.data_lock:
                    self.sock.sendto(connection.build_receiver_packet(FIN_FLAG), connection.src_address)
            except OSError:
                continue
        self.open = False
        return

    # background thread that deals with pakets as they come in
    def packet_handler(self):
        i = 0
        # print("packet_handler running")
        while self.open:
            # print("waiting for packet")
            udp_packet = None
            try:
                udp_packet = self.sock.recvfrom(MAX_PACKET) # change back to recv later?
            except socket.timeout:
                continue
            # print("received packet")
            # print(udp_packet[0])
            if not self.verify_crc(udp_packet[0]):
                # print("An error in the packet. Dropped " + str(i))
                # print(udp_packet[0])
                i += 1
                continue
            # print("packet received from: " + str(udp_packet[1]))
            # with self.data_lock:
            packet = SenderPacket(udp_packet[0], udp_packet[1])
            packet_address = udp_packet[1]
            response_packet = ReceiverPacket(udp_packet[0])
            if packet.flags == SYN_FLAG and packet.data == None:
                # print("received a syn_request at " + str(packet.src_address))
                with self.data_lock:
                    self.create_connection(packet)
            elif packet.flags == CLI_ACK_FLAG: 
                if self.pending_connections_dict.get(packet.src_address) != None and packet.data == None:
                    with self.data_lock:
                        self.handle_ack(self.pending_connections_dict[packet.src_address], packet)
            elif packet.flags == 0:
                if self.in_connections.get(packet.src_address) != None:
                    with self.data_lock:
                        self.handle_packet(self.in_connections[packet.src_address], packet)
                elif self.pending_connections_dict.get(packet.src_address) != None:
                    with self.data_lock:
                        self.handle_packet(self.pending_connections_dict[packet.src_address], packet)
                else:
                    print("did not enter either case")
            elif packet.flags == DIS_FLAG:
                if self.in_connections.get(packet.src_address) != None:
                    with self.data_lock:
                        self.handle_cls(self.in_connections[packet.src_address])
                elif self.pending_connections_dict.get(packet.src_address) != None:
                    with self.data_lock:
                        self.handle_cls(self.pending_connections_dict[packet.src_address])
            elif self.out_connections.get(packet_address) != None:
                with self.out_connections[packet_address].packet_lock:
                    self.out_connections[packet_address].packet_buffer.append(response_packet)
                # print("received syn-ack")
            else:
                print("entered no case")
                print(self.out_connections)
      
        # print("Server/Receiver is closed")
        self.sock.close()
        return None
    



    def create_connection(self, packet):
        if self.in_connections.get(packet.src_address) != None: # this is a live connection, cant accept anything here
            return
        if  self.pending_connections_dict.get(packet.src_address) != None: 
            if self.pending_connections_dict[packet.src_address].connected == False: # likely this syn ack got lost send it again # CHECK THAT SEQUENCE NUMBER MATCHES
                self.sock.sendto(self.pending_connections_dict[packet.src_address].build_receiver_packet(SYN_FLAG | ACK_FLAG), packet.src_address)
                # print("resend syn ack to " + str(packet.src_address) + "packet seq  number: " + str(packet.sequence_number) + "connection sequence number: " + str(self.pending_connections_dict[packet.src_address].sequence_number))
            return
        # create the new connection and send a syn ack
        connection = ReceiverConnection(packet.src_address, packet.sequence_number + 1, self.data_lock)
        self.pending_connections_dict[packet.src_address] = connection
        self.sock.sendto(connection.build_receiver_packet(SYN_FLAG | ACK_FLAG), packet.src_address)
        # print("send syn_ack to " + str(packet.src_address) + "packet seq  number: " + str(packet.sequence_number) + "connection sequence number: " + str(self.pending_connections_dict[packet.src_address].sequence_number))
        return


    # handle ack in the handshake
    def handle_ack(self, connection, packet):
        sequence_num = packet.sequence_number
        if sequence_num != connection.sequence_number:
            return
        if connection.src_address not in self.ready_addresses:
            with self.connect_lock:
                self.ready_connections.append(connection)
                self.ready_addresses.add(connection.src_address)

    # handle all data packets
    def handle_packet(self, connection, packet):
        if connection.connected == False:

            if connection.src_address not in self.ready_addresses:
                # print("adding connection at " + str(connection.src_address) + "to ready connections and addresses")
                self.ready_addresses.add(connection.src_address)
                self.ready_connections.append(connection)
            return

        else:
            last_sequence_number = packet.sequence_number + packet.length - 1
            # print("\nown seq. number: " + str(connection.sequence_number) + "\npacket seq. number: " + str(packet.sequence_number) + "\nlast_seq_number: " + str(last_sequence_number))

            max_sequence_in_window = (connection.sequence_number + connection.window_size) 
            if last_sequence_number > max_sequence_in_window:
                print("packet out of bounds")
                return
            if packet.data == None:
                print("an empty packet")
            if packet.sequence_number >= connection.sequence_number and packet.data != None:

                # print("\nreceived sequence number " + str(packet.sequence_number) + "\n")
                connection.window[packet.sequence_number - connection.sequence_number: 1 + last_sequence_number - connection.sequence_number] = packet.data

                if packet.sequence_number == connection.sequence_number: # this packet is the beginning of window, we can send everything out to go
                    # the second check above is to make sure we don't put things in the outbuffer
                    curr_end = last_sequence_number
                    while connection.ranges.get(curr_end + 1) != None:
                        curr_end = connection.ranges.pop(curr_end + 1)


                    add_to_outbuffer = connection.window[:1 + curr_end - connection.sequence_number]

                    if len(connection.outbuffer) < MAX_OUTBUFFER: # room in outbuffer
                        if len(connection.outbuffer) + len(add_to_outbuffer) <= MAX_OUTBUFFER: # for all consecutive data
                            connection.window = connection.window[curr_end - connection.sequence_number + 1:]
                            connection.sequence_number = curr_end + 1
                        else: # room for some data
                            qty_to_outbuffer = MAX_OUTBUFFER - len(connection.outbuffer)
                            add_to_outbuffer = connection.window[:qty_to_outbuffer]
                            connection.window = connection.window[qty_to_outbuffer:]
                            connection.ranges[connection.sequence_number + qty_to_outbuffer] = curr_end
                            connection.sequence_number = connection.sequence_number + qty_to_outbuffer
                        connection.outbuffer.extend(add_to_outbuffer)
                        add_to_window_restore = connection.window_size - len(connection.window)
                        connection.window.extend(([0]*add_to_window_restore))

                    connection.timestamp = time.time()
                else:
                    connection.ranges[packet.sequence_number] = last_sequence_number
                    if connection.timestamp - time.time() <= TIMEOUT or len(connection.outbuffer) >= MAX_OUTBUFFER:
                        return
            try:
                self.sock.sendto(connection.build_receiver_packet(ACK_FLAG) , packet.src_address)
                # print("sending ack to " + str(connection.src_address) + "at sequence number: " + str(connection.sequence_number))               
            except OSError:
                return
            return

    # handle a disconnect message
    def handle_cls(self, connection):
        self.sock.sendto(connection.build_receiver_packet(ACK_FLAG | DIS_FLAG), connection.src_address)
        connection.connected = False

    
    def mrt_connect(self, dest_ip, dest_port):
        # print("calling mrt_connect")
        # print(self.out_connections)
        if self.out_connections.get((dest_ip, dest_port)) != None:
            print("Already connected here!")
            return None
        connection = SenderConnection(self.sock, dest_ip, dest_port, self.data_lock)
        self.out_connections[(dest_ip, dest_port)] = connection
        # connection_thread = threading.Thread(target=connection.connect, args=())
        # connection_thread.start()
        connection.connect()
        # print("connect returned")

        # print("Connected with sequence number " + str(connection.sequence_number))
        return connection

    

class ReceiverConnection:
    def __init__(self, src_address, sequence_number, data_lock):
        self.src_address = src_address
        self.connected = False
        self.sequence_number = sequence_number
        self.window_size = RECEIVER_WINDOW # in bytes maybe make thissmaller
        self.window = bytearray(([0]*self.window_size))
        self.outbuffer = bytearray() 
        self.ranges = dict() # map beginning of range to end of range
        self.timestamp = time.time()
        self.data_lock = data_lock

    @staticmethod
    def mrt_probe(connections):
        return_connection = None
        # with lock:
        for connection in connections:
            if len(connection.outbuffer) != 0:
                return_connection = connection
                break
        return return_connection

    def mrt_receive1(self):
        data = []
        while len(data) == 0:
            while len(self.outbuffer) == 0:
                time.sleep(0.04)
                if self.connected == False:
                    print("not actually connected")
                    return
            with self.data_lock:
                len_data = len(self.outbuffer)
                data = self.outbuffer[:len_data]
                # self.outbuffer = self.outbuffer[len_data:]
                self.outbuffer = bytearray()
        return data

     # helper function to construct packets     
    def build_receiver_packet(self, flags):
        ack_num_7 = self.sequence_number >> 56
        ack_num_6 = (self.sequence_number >> 48) & 0x00FF
        ack_num_5 = (self.sequence_number >> 40) & 0x0000FF
        ack_num_4 = (self.sequence_number >> 32) & 0x000000FF
        ack_num_3 = (self.sequence_number >> 24) & 0x00000000FF
        ack_num_2 = (self.sequence_number >> 16) & 0x0000000000FF
        ack_num_1 = (self.sequence_number >> 8) & 0x000000000000FF
        ack_num_0 = self.sequence_number & 0x00000000000000FF
        window_size_1 = self.window_size >> 8
        window_size_0 = self.window_size & 0x00FF
        checksum = zlib.crc32(bytes([ack_num_7, ack_num_6, ack_num_5, ack_num_4, ack_num_3, ack_num_2, ack_num_1, ack_num_0, flags,window_size_1, window_size_0]))
        checksum_3 = checksum >> 24
        checksum_2 = (checksum >> 16) & 0x00FF
        checksum_1 = (checksum >> 8) & 0x0000FF
        checksum_0 = checksum & 0x000000FF
        # checksum_0 = (checksum_0 + 1) % 256
        return bytes([checksum_3, checksum_2, checksum_1, checksum_0, ack_num_7, ack_num_6, ack_num_5, ack_num_4, ack_num_3, ack_num_2, ack_num_1, ack_num_0, flags, window_size_1, window_size_0])

class SenderConnection:
    def __init__(self, sock, dest_ip, dest_port, data_lock):
        self.sock = sock
        self.dest_port = dest_port
        self.dest_address = (dest_ip, dest_port)
        self.packet_buffer = []
        self.connected = False
        self.sequence_number = random.randint(0, 4095)
        self.server_window_size = 0
        self.sender_window_size = INIT_SENDER_WINDOW 
        self.data_packet_size = PACKET_SIZE
        self.data_lock = data_lock
        self.packet_lock = threading.Lock()
       


    def mrt_send(self, data):
        bytes_to_send = len(data)
        total_bytes_ackd = 0
        num_packets = bytes_to_send // (self.data_packet_size)
        if bytes_to_send % self.data_packet_size != 0:
            num_packets += 1
        bytes_sent_in_window = 0
        while total_bytes_ackd < bytes_to_send:
            effective_window = min(self.server_window_size, self.sender_window_size)
            while bytes_sent_in_window < effective_window and total_bytes_ackd + bytes_sent_in_window < bytes_to_send: # later make this a min of server and sender window size ##and total_bytes_ackd + bytes_sent_in_window < bytes_to_send
                data_upper_bound = min(total_bytes_ackd + bytes_sent_in_window + self.data_packet_size, len(data))
                packet_data = data[total_bytes_ackd+bytes_sent_in_window: data_upper_bound]
                temp_sequence_number = self.sequence_number + bytes_sent_in_window
                packet = self.build_data_packet(temp_sequence_number, packet_data)
                with self.data_lock:
                    self.sock.sendto(packet, self.dest_address)
                bytes_sent_in_window += len(packet_data)
                
            # print("Sent all bytes in window: " + str(bytes_sent_in_window))
            temp_ack_number = self.sequence_number

            time.sleep(TIMEOUT)
            with self.packet_lock:
                while len(self.packet_buffer) > 0:
                    receiver_packet = self.packet_buffer.pop(0)
                    
                    # if not self.verify_crc(receiver_packet.raw_packet):
                    #     print("crc error")
                    #     continue
                    if receiver_packet.flags != ACK_FLAG:
                        if receiver_packet.flags == FIN_FLAG:
                            print("Error: the server is closed") # maybe call disconnect here?
                            return
                        continue
                    # print("ack received! at seq. number " + str(receiver_packet.ack_number))
                    ack_num = receiver_packet.ack_number
                    if ack_num > temp_ack_number:
                        temp_ack_number = ack_num
                        # maybe not put these in if case
                        new_window = receiver_packet.window_size
                        self.server_window_size = new_window
            

            total_bytes_ackd += temp_ack_number - self.sequence_number 
            if temp_ack_number == self.sequence_number: # we didnt receive any ack so shrink the sender window
                self.sender_window_size = max(self.data_packet_size, (((self.sender_window_size // 2) // self.data_packet_size) * self.data_packet_size)) # ensure window is a multiple of packets
            elif (temp_ack_number - self.sequence_number) == effective_window:
                # print("Increasing window size")
                self.sender_window_size += self.data_packet_size
            # print("sender_window: " + str(self.sender_window_size))
            self.sequence_number = temp_ack_number
            bytes_sent_in_window = 0
        # print("sent successfully")
        return

    def mrt_disconnect(self):
        success = False
        first = time.time()
        while not success and time.time() - first < 15: # we'll try for 15 seconds before giving up
            # response = None
            with self.data_lock:
                self.sock.sendto(self.build_no_data_packet(DIS_FLAG), self.dest_address)
            # try:
            #     response = self.sock.recv(INIT_PACKET)
            # except socket.timeout:
            #     continue
            time.sleep(TIMEOUT)
            response_packet = None
            with self.packet_lock:
                if len(self.packet_buffer) == 0:
                    continue
                else:
                    response_packet = self.packet_buffer.pop(0)
            # response_packet = ReceiverPacket(response)
            if not self.verify_crc(response_packet.raw_packet):
                print("A bad packet!")
            elif response_packet.flags != (ACK_FLAG | DIS_FLAG):
                if response_packet.flags == FIN_FLAG:
                    print("The server is already closed")
                    success = True
                else:
                    print("An unknown flag")
            else:
                success = True
        self.connected = False




    def verify_crc(self, packet):
        packet_crc = (packet[0] << 24) + (packet[1] << 16) + (packet[2] << 8) + (packet[3])
        return zlib.crc32(packet[SEQUENCE_NUM_7:]) == packet_crc
    
    
    # helper function called by connect
    def connect(self):
        while not self.connected:
            # response = None
            # print("sending syn message")
            with self.data_lock:
                self.sock.sendto(self.build_no_data_packet(SYN_FLAG), self.dest_address)
            time.sleep(TIMEOUT)
            response_packet = None
            with self.packet_lock:
                if len(self.packet_buffer) == 0:
                    continue
                response_packet = self.packet_buffer.pop(0)

            # try:
            #     response = self.sock.recv(INIT_PACKET)
            # except socket.timeout:
            #     continue

            # response_packet = ReceiverPacket(response)
            if not self.verify_crc(response_packet.raw_packet):
                print("bad CRC")
            elif response_packet.flags != (SYN_FLAG | ACK_FLAG):
                print("not a syn-ack")
                if response_packet.flags == FIN_FLAG:
                    print("server is closed")
                    self.connected = True
            elif response_packet.ack_number != self.sequence_number + 1:
                print("error: unexpected sequence number. Packet: " + str(response_packet.ack_number) + " own seq num: " + str(self.sequence_number))
            else:
                # print("hooray we are connecting:")
                self.sequence_number += 1
                self.server_window_size = response_packet.window_size
                with self.data_lock:
                    self.sock.sendto(self.build_no_data_packet(CLI_ACK_FLAG), self.dest_address)
                # print("sent the ack")
                self.connected = True
            


     # two helper functions to construct data        
    def build_no_data_packet(self, flags):
        sequence_num_7 = self.sequence_number >> 56
        sequence_num_6 = (self.sequence_number >> 48) & 0x00FF
        sequence_num_5 = (self.sequence_number >> 40) & 0x0000FF
        sequence_num_4 = (self.sequence_number >> 32) & 0x000000FF
        sequence_num_3 = (self.sequence_number >> 24) & 0x00000000FF
        sequence_num_2 = (self.sequence_number >> 16) & 0x0000000000FF
        sequence_num_1 = (self.sequence_number >> 8) & 0x000000000000FF
        sequence_num_0 = self.sequence_number & 0x00000000000000FF
        length_1 = 0
        length_0 = 0
        checksum = zlib.crc32(bytes([sequence_num_7, sequence_num_6, sequence_num_5, sequence_num_4, sequence_num_3, sequence_num_2, sequence_num_1, sequence_num_0, flags,length_1, length_0]))
        checksum_3 = checksum >> 24
        checksum_2 = (checksum >> 16) & 0x00FF
        checksum_1 = (checksum >> 8) & 0x0000FF
        checksum_0 = checksum & 0x000000FF
        return bytes([checksum_3, checksum_2, checksum_1, checksum_0,sequence_num_7, sequence_num_6, sequence_num_5, sequence_num_4, sequence_num_3, sequence_num_2, sequence_num_1, sequence_num_0, flags,length_1, length_0])

    def build_data_packet(self, temp_sequence_number, data):
        sequence_num_7 = temp_sequence_number >> 56
        sequence_num_6 = (temp_sequence_number >> 48) & 0x00FF
        sequence_num_5 = (temp_sequence_number >> 40) & 0x0000FF
        sequence_num_4 = (temp_sequence_number >> 32) & 0x000000FF
        sequence_num_3 = (temp_sequence_number >> 24) & 0x00000000FF
        sequence_num_2 = (temp_sequence_number >> 16) & 0x0000000000FF
        sequence_num_1 = (temp_sequence_number >> 8) & 0x000000000000FF
        sequence_num_0 = temp_sequence_number & 0x00000000000000FF
        flags = 0 
        length_1 = len(data) >> 8
        length_0 = len(data) & 0x00FF
        packet = bytearray([0,0,0,0,sequence_num_7, sequence_num_6, sequence_num_5, sequence_num_4, sequence_num_3, sequence_num_2, sequence_num_1, sequence_num_0, flags, length_1, length_0])
        packet.extend(data)
        checksum = zlib.crc32(packet[SEQUENCE_NUM_7:])
        packet[CHECKSUM_3] = checksum >> 24
        packet[CHECKSUM_2] = (checksum >> 16) & 0x00FF
        packet[CHECKSUM_1] = (checksum >> 8) & 0x0000FF
        packet[CHECKSUM_0] = checksum & 0x000000FF
        return packet



    

# two helper classes to facilitate access to packets
class ReceiverPacket:
    def __init__ (self, raw_packet):
        
        self.checksum = (raw_packet[CHECKSUM_3] << 24) + (raw_packet[CHECKSUM_2] << 16) + (raw_packet[CHECKSUM_1] << 8) + raw_packet[CHECKSUM_0]

        self.ack_number = ((raw_packet[SEQUENCE_NUM_7] << 56) + (raw_packet[SEQUENCE_NUM_6] << 48) + (raw_packet[SEQUENCE_NUM_5] << 40)
            + (raw_packet[SEQUENCE_NUM_4] << 32) + (raw_packet[SEQUENCE_NUM_3] << 24) + (raw_packet[SEQUENCE_NUM_2] << 16)
            + (raw_packet[SEQUENCE_NUM_1] << 8) + (raw_packet[SEQUENCE_NUM_0]))

        self.flags = raw_packet[FLAG]
        self.window_size = (raw_packet[WINDOW_SIZE_1] << 8) + raw_packet[WINDOW_SIZE_0]
        self.raw_packet = raw_packet # for debug


class SenderPacket:
    def __init__ (self, raw_packet, src_address):
        self.src_address = src_address
        self.checksum = (raw_packet[CHECKSUM_3] << 24) + (raw_packet[CHECKSUM_2] << 16) + (raw_packet[CHECKSUM_1] << 8) + raw_packet[CHECKSUM_0]
        self.sequence_number = ((raw_packet[SEQUENCE_NUM_7] << 56) + (raw_packet[SEQUENCE_NUM_6] << 48) + (raw_packet[SEQUENCE_NUM_5] << 40)
            + (raw_packet[SEQUENCE_NUM_4] << 32) + (raw_packet[SEQUENCE_NUM_3] << 24) + (raw_packet[SEQUENCE_NUM_2] << 16)
            + (raw_packet[SEQUENCE_NUM_1] << 8) + (raw_packet[SEQUENCE_NUM_0]))

        self.flags = raw_packet[FLAG]
        self.length = (raw_packet[LEN_1] << 8) + raw_packet[LEN_0]
        self.data = None
        if len(raw_packet) > HEADER_LEN:
            self.data = raw_packet[HEADER_LEN:]
        self.raw_packet = raw_packet # for debug purposes
