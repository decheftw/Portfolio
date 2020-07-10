import socket
import random
import time
import zlib


'''
Sender to Receiver:
Don't need SourcePort(16) else would get messed up in NAT, just use udp src port
CheckSum(32)
Sequence#(64)
Flags(8) (syn,ack,fin,cls,0,0,0,0)
Length(16)

Data()

Receiver to Sender:
Checksum(32)
Ack#(64)
Flags(8) (syn,ack,cls,fin,0,0,0,0)
Windowsize(16)

'''
TIMEOUT = 0.4 # 400 milliseconds
INIT_PACKET = 128
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
LEN_1 = 13 # do we even need a length field if its packaged up in the udp packet, could just do len(packet)
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

class ReceiverPacket:
    def __init__ (self, raw_packet):
        
        self.checksum = (raw_packet[CHECKSUM_3] << 24) + (raw_packet[CHECKSUM_2] << 16) + (raw_packet[CHECKSUM_1] << 8) + raw_packet[CHECKSUM_0]

        self.ack_number = ((raw_packet[SEQUENCE_NUM_7] << 56) + (raw_packet[SEQUENCE_NUM_6] << 48) + (raw_packet[SEQUENCE_NUM_5] << 40)
            + (raw_packet[SEQUENCE_NUM_4] << 32) + (raw_packet[SEQUENCE_NUM_3] << 24) + (raw_packet[SEQUENCE_NUM_2] << 16)
            + (raw_packet[SEQUENCE_NUM_1] << 8) + (raw_packet[SEQUENCE_NUM_0]))

        # (raw_packet[SEQUENCE_NUM_1] << 8) + raw_packet[SEQUENCE_NUM_0]
        self.flags = raw_packet[FLAG]
        self.window_size = (raw_packet[WINDOW_SIZE_1] << 8) + raw_packet[WINDOW_SIZE_0]
        self.raw_packet = raw_packet

class SenderConnection:
    def __init__(self, sock, dest_ip, dest_port):
        self.sock = sock
        # self.src_port = dest_port
        self.dest_port = dest_port
        self.dest_address = (dest_ip, dest_port)
        self.connected = False
        self.sequence_number = random.randint(0, 4095)
        self.server_window_size = 0
        self.sender_window_size = 256 
        self.data_packet_size = 64

    # maybe not ask for a src port but just pick one your self
    @staticmethod
    def mrt_connect(dest_ip, dest_port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((dest_ip, 0))
            sock.settimeout(TIMEOUT)        
        except OSError:
            print("connect_error: unable to connect at provided source port")
            return None
        connection = SenderConnection(sock, dest_ip, dest_port)
        connection.connect()
        print("Connected with sequence number " + str(connection.sequence_number))
        return connection

    @staticmethod
    def calc_crc(data):
        return zlib.crc32(data)
    
    def verify_crc(self, packet):
        packet_crc = (packet[0] << 24) + (packet[1] << 16) + (packet[2] << 8) + (packet[3])
        return zlib.crc32(packet[SEQUENCE_NUM_7:]) == packet_crc
    
    

    def connect(self):
        while not self.connected:
            response = None
            print("sending syn message")
            self.sock.sendto(self.build_no_data_packet(SYN_FLAG), self.dest_address)
            try:
                response = self.sock.recv(INIT_PACKET)
            except socket.timeout:
                continue
            response_packet = ReceiverPacket(response)
            if not self.verify_crc(response):
                print("bad CRC")
            elif response_packet.flags != (SYN_FLAG | ACK_FLAG):
                print("not a syn-ack")
                if response_packet.flags == FIN_FLAG:
                    print("server is closed")
                    self.connected = True
            elif response_packet.ack_number != self.sequence_number + 1:
                print("error: unexpected sequence number. Packet: " + str(response_packet.ack_number) + " own seq num: " + str(self.sequence_number))
            else:
                print("hooray we are connecting:")
                print("packet " + str(response))
                self.sequence_number += 1
                self.server_window_size = response_packet.window_size
                self.sock.sendto(self.build_no_data_packet(ACK_FLAG), self.dest_address)
                print("sent the ack")
                self.connected = True
            

             
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
        # print(packet)
        return packet

    
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
                # print("effective window: " + str(effective_window))
                packet_data = data[total_bytes_ackd+bytes_sent_in_window: data_upper_bound]
                # print("packet_data: " + packet_data.decode("utf8"))
                temp_sequence_number = self.sequence_number + bytes_sent_in_window
                packet = self.build_data_packet(temp_sequence_number, packet_data)
                self.sock.sendto(packet, self.dest_address)
                bytes_sent_in_window += len(packet_data)
                
            print("Sent all bytes in window: " + str(bytes_sent_in_window))
            temp_ack_number = self.sequence_number
            while True:
                try:
                    ack = self.sock.recv(INIT_PACKET)
                    receiver_packet = ReceiverPacket(ack)
                    if not self.verify_crc(ack):
                        continue
                    if ack[FLAG] != ACK_FLAG:
                        if ack[FLAG] == FIN_FLAG:
                            print("Error: the server is closed")
                            return
                        continue
                    print("ack received! at seq. number " + str(receiver_packet.ack_number))
                    ack_num = receiver_packet.ack_number
                    if ack_num > temp_ack_number:
                        temp_ack_number = ack_num
                        # maybe not put these in if case
                        new_window = receiver_packet.window_size
                        self.server_window_size = new_window
                except socket.timeout:
                    break
                # handle checksum bytes same as below, continue if incorrect

            total_bytes_ackd += temp_ack_number - self.sequence_number 
            if temp_ack_number == self.sequence_number: # we didnt receive any ack so shrink the sender window
                self.sender_window_size = max(self.data_packet_size, (((self.sender_window_size // 2) // self.data_packet_size) * self.data_packet_size)) # ensure window is a multiple of packets
            elif (temp_ack_number - self.sequence_number) == effective_window:
                print("incerasing data size")
                self.sender_window_size += self.data_packet_size
            # print("sender_window: " + str(self.sender_window_size))
            self.sequence_number = temp_ack_number
            
           
            bytes_sent_in_window = 0
            # time.sleep(2)
        print("sent successfully")
        return

    def mrt_disconnect(self):
        # close_message = self.build_no_data_packet(DIS_FLAG)
        # cls_resp = None 
        # self.sock.sendto(close_message, self.dest_address) # maybe
        success = False
        first = time.time()
        while not success and time.time() - first < 15: # we'll try for 15 seconds before giving up
            response = None
            self.sock.sendto(self.build_no_data_packet(DIS_FLAG), self.dest_address)
            try:
                response = self.sock.recv(INIT_PACKET)
            except socket.timeout:
                continue
            response_packet = ReceiverPacket(response)
            if not self.verify_crc(response):
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
        self.sock.close()


