# Shikhar Sinha
# Professor Joosten
# COSC 60 Computer Networks Spring 2020
# Lab 3: Mini Reliable Transport
import socket
import threading
import time
import zlib
'''
TODO:
PUT ALL IN ONE FILE????
Remove Redundancies

'''
# use host as '' to accept INADDR_ANY
DEFAULT_PORT = 22222 # default port
MAX_PACKET = 1200 # for now, subject to change
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
MAX_OUTBUFFER = 256

lock = threading.Lock()
# Class to make access to the packet data easier. Less need to index into the returned packet
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
        self.raw_packet = raw_packet
        # if len packet less than header len, return null?

class ServerConnection:
    def __init__(self, src_address, sequence_number):
        self.src_address = src_address
        self.connected = False
        self.sequence_number = sequence_number
        self.window_size = 256 # in bytes maybe make thissmaller
        self.window = bytearray(([0]*self.window_size))
        self.outbuffer = bytearray() # how to initialize this one hmm...
        self.ranges = dict() # map beginning of range to end of range
        self.timestamp = time.time()

    @staticmethod
    def mrt_probe(connections):
        return_connection = None
        with lock:
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
            with lock:
                len_data = len(self.outbuffer)
                data = self.outbuffer[:len_data]
                self.outbuffer = self.outbuffer[len_data:]
        return data
          
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
        return bytes([checksum_3, checksum_2, checksum_1, checksum_0, ack_num_7, ack_num_6, ack_num_5, ack_num_4, ack_num_3, ack_num_2, ack_num_1, ack_num_0, flags, window_size_1, window_size_0])
    
    



class Server:

    def __init__(self, sock, port):
        self.sock = sock
        self.port = port
        self.ready_connections = list()
        self.ready_addresses = set()
        self.pending_connections_dict = dict() # key = address
        self.connections = dict() # index a connection object by its port,ip combo
        self.open = True

    def verify_crc(self, packet):
        packet_crc = (packet[0] << 24) + (packet[1] << 16) + (packet[2] << 8) + (packet[3])
        return zlib.crc32(packet[SEQUENCE_NUM_7:]) == packet_crc
    # mrt_open: indicate ready-ness to receive incoming connections
    @staticmethod
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
            print("opening server at port" + str(DEFAULT_PORT))
            server = Server(sock, DEFAULT_PORT)
            packet_thread = threading.Thread(target=server.packet_handler, args=())
            packet_thread.start()
            print("server started at default port")
            return server
        sock.settimeout(2)
        server = Server(sock, port)
        packet_thread = threading.Thread(target=server.packet_handler, args=())
        packet_thread.start()
        print("server started at input port")
        return server

    def packet_handler(self):
        i = 0
        print("packet_handler running")
        while self.open:
            # print("waiting for pakcet")
            udp_packet = None
            try:
                udp_packet = self.sock.recvfrom(MAX_PACKET) # change back to recv later?
            except socket.timeout:
                continue

            if not self.verify_crc(udp_packet[0]):
                # print("An error in the packet. Dropped " + str(i))
                # print(udp_packet[0])
                i += 1
                continue
            # print("packet received from: " + str(udp_packet[1]))
            with lock:
                packet = SenderPacket(udp_packet[0], udp_packet[1])
                if packet.flags == SYN_FLAG and packet.data == None:
                    # print("received a syn_request at " + str(packet.src_address))
                    self.create_connection(packet)
                elif packet.flags == ACK_FLAG: 
                    # print("second ack received " + str(packet.src_address) + "seq num: " + str(packet.sequence_number))
                    if self.pending_connections_dict.get(packet.src_address) != None and packet.data == None:
                        # print("handling ack for " + str(packet.src_address) + "seq num: " + str(packet.sequence_number))
                        self.handle_ack(self.pending_connections_dict[packet.src_address], packet)
                elif packet.flags == 0:
                    # print("handle a data packet for " + str(packet.src_address))
                    if self.connections.get(packet.src_address) != None:
                        # print(str(packet.src_address) + "is in connections")
                        self.handle_packet(self.connections[packet.src_address], packet)
                    elif self.pending_connections_dict.get(packet.src_address) != None:
                        # print(str(packet.src_address) + "is in pending connections")
                        self.handle_packet(self.pending_connections_dict[packet.src_address], packet)
                    else:
                        print("did not enter either case")

                elif packet.flags == DIS_FLAG:
                    if self.connections.get(packet.src_address) != None:
                        self.handle_cls(self.connections[packet.src_address])
                    elif self.pending_connections_dict.get(packet.src_address) != None:
                        self.handle_cls(self.pending_connections_dict[packet.src_address])
                # for connection in pending_connections_dict.values 
      


        print("Server/Receiver is closed")
        self.sock.close()
        return None

    '''
    A SYN Packet came in.
    1. Check if we already have an established connection at this address. If we do, ignore this packet
    2. Check if we have already seen a SYN Packet at this address (but not accepted a connection) If this
    is the case, the SYN-ACK got lost so we resent a SYN-ACK.
    3. 
    '''
    def create_connection(self, packet):
        if self.connections.get(packet.src_address) != None: # this is a live connection, cant accept anything here
            return
        if  self.pending_connections_dict.get(packet.src_address) != None: 
            if self.pending_connections_dict[packet.src_address].connected == False: # likely this syn ack got lost send it again # CHECK THAT SEQUENCE NUMBER MATCHES
                self.sock.sendto(self.pending_connections_dict[packet.src_address].build_receiver_packet(SYN_FLAG | ACK_FLAG), packet.src_address)
                # print("resend syn ack to " + str(packet.src_address) + "packet seq  number: " + str(packet.sequence_number) + "connection sequence number: " + str(self.pending_connections_dict[packet.src_address].sequence_number))
            return
        # create the new connection and send a syn ack
        connection = ServerConnection(packet.src_address, packet.sequence_number + 1)
        self.pending_connections_dict[packet.src_address] = connection
        self.sock.sendto(connection.build_receiver_packet(SYN_FLAG | ACK_FLAG), packet.src_address)
        # print("send syn_ack to " + str(packet.src_address) + "packet seq  number: " + str(packet.sequence_number) + "connection sequence number: " + str(self.pending_connections_dict[packet.src_address].sequence_number))
        return




    def mrt_accept1(self):
        while(len(self.ready_connections) == 0):
            time.sleep(1) # maybe do this in a better way
        with lock: #race condition
            connection = self.ready_connections.pop(0)
            self.ready_addresses.remove(connection.src_address)
            self.pending_connections_dict.pop(connection.src_address)
            connection.connected = True
            self.connections[connection.src_address] = connection
            print("Connected! Sequence Number is " + str(connection.sequence_number) + ", src address is " + str(connection.src_address))
        return connection
    
    def mrt_accept_all(self):
        new_connections = []
        with lock:
            num_available_connections = len(self.ready_connections)
            i = 0
            while i < num_available_connections:
                connection = self.ready_connections.pop(0)
                self.ready_addresses.remove(connection.src_address)
                self.pending_connections_dict.pop(connection.src_address)
                new_connections.append(connection)
                connection.connected = True
                i += 1
        return new_connections
         


    def handle_ack(self, connection, packet):
        sequence_num = packet.sequence_number
        if sequence_num != connection.sequence_number:
            return
        if connection.src_address not in self.ready_addresses:
            self.ready_connections.append(connection)
            self.ready_addresses.add(connection.src_address)
            print("adding connection at " + str(connection.src_address) + "to ready connections and addresses")


    def handle_packet(self, connection, packet):
        if connection.connected == False:
            print("not yet connected! at " + str(connection.src_address)) # we need to handle when the 3rd handshake gets lost and they just start sending data

            if connection.src_address not in self.ready_addresses:
                print("adding connection at " + str(connection.src_address) + "to ready connections and addresses")
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
                # self.sock.sendto(connection.build_receiver_packet(ACK_FLAG), connection.src_address)
                # print("sending ack to " + str(connection.src_address) + "at sequence number: " + str(connection.sequence_number))
                # return
            # if packet.sequence_number < connection.sequence_number:
            #     self.sock.sendto(connection.build_receiver_packet(ACK_FLAG), connection.src_address)
            #     print("sending ack to " + str(connection.src_address) + "at sequence number: " + str(connection.sequence_number))
            if packet.sequence_number >= connection.sequence_number and packet.data != None:


                connection.window[packet.sequence_number - connection.sequence_number: 1 + last_sequence_number - connection.sequence_number] = packet.data

                if packet.sequence_number == connection.sequence_number: # this packet is the beginning of window, we can send everything out to go
                    # the second check above is to make sure we don't put things in the outbuffer
                    curr_end = last_sequence_number
                    while connection.ranges.get(curr_end + 1) != None:
                        curr_end = connection.ranges.pop(curr_end + 1)


                    add_to_outbuffer = connection.window[:1 + curr_end - connection.sequence_number]

                    if len(connection.outbuffer) < MAX_OUTBUFFER:
                        if len(connection.outbuffer) + len(add_to_outbuffer) <= MAX_OUTBUFFER:
                            # connection.outbuffer.extend(add_to_outbuffer)
                            connection.window = connection.window[curr_end - connection.sequence_number:]
                            # add_to_window_restore = connection.window_size - len(connection.window)
                            # connection.window.extend(([0]*add_to_window_restore))
                            connection.sequence_number = curr_end + 1
                            # self.sock.sendto(connection.build_receiver_packet(ACK_FLAG) , packet.src_address)
                        else:
                            qty_to_outbuffer = MAX_OUTBUFFER - len(connection.outbuffer)
                            add_to_outbuffer = connection.window[:qty_to_outbuffer]
                            connection.window = connection.window[qty_to_outbuffer:]
                            connection.ranges[connection.sequence_number + qty_to_outbuffer] = curr_end
                            connection.sequence_number = connection.sequence_number + qty_to_outbuffer
                        connection.outbuffer.extend(add_to_outbuffer)
                        add_to_window_restore = connection.window_size - len(connection.window)
                        connection.window.extend(([0]*add_to_window_restore))
                    # self.sock.sendto(connection.build_receiver_packet(ACK_FLAG) , packet.src_address)
                    # print("sending ack to " + str(connection.src_address) + "at sequence number: " + str(connection.sequence_number))
                    connection.timestamp = time.time()
                else:
                    connection.ranges[packet.sequence_number] = last_sequence_number
                    if connection.timestamp - time.time() <= TIMEOUT or len(connection.outbuffer) >= MAX_OUTBUFFER:
                        return

            self.sock.sendto(connection.build_receiver_packet(ACK_FLAG) , packet.src_address)
            # print("sending ack to " + str(connection.src_address) + "at sequence number: " + str(connection.sequence_number))               
            return

    def handle_cls(self, connection):
        self.sock.sendto(connection.build_receiver_packet(ACK_FLAG | DIS_FLAG), connection.src_address)
        connection.connected = False

    def mrt_close(self):
        with lock:
            self.open = False
            for connection in self.connections.values():
                self.sock.sendto(connection.build_receiver_packet(FIN_FLAG), connection.src_address)
            for connection in self.pending_connections_dict.values():
                self.sock.sendto(connection.build_receiver_packet(FIN_FLAG), connection.src_address)
            # self.sock.close()
            print("successfully closed")
        return















