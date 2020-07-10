from threading import *
from struct import *
import time
close = False
import threading
import sys
from socket import *
import signal
# 4 bytes of checksum, 4 bytes of packet type, 2 bytes of fragment number, 4 bytes of connection identifiier, 4 bytes of window
# RCON = Request to connect
# AKRQ = Acknowledge Request to connect
# DATA = Data message
# ADAT = Acknowledge Data
# RQCL = Request to close
# ACLS = Acknowledge close request
# CLOS = Close
# BADD = corrupted packet

# format_string = "l4sii980s"
connection = ("127.0.0.1", 5005)
window_size = 5

lock = threading.Lock()

class Packet:
    def __init__(self, packet, address):
        self.packet = packet
        self.address = address


class Connection:
    def __init__(self, address, connected):
        self.address = address
        self.connected = connected
        self.buffers = []
        self.received = []


class Server(Thread):
    def __init__(self, sock):
        Thread.__init__(self)
        self.sock = sock
        self.requests = []
        self.connects = []
        self.close = False

    def run(self):   # Background process that receives packets for the receiver and processes them
        i = 1
        while self.close == False:
            data, address = self.sock.recvfrom(1000)
            lock.acquire(blocking=1)
            string = buffer_reader(data)            # unpacks the buffer
            if checksum_validate(string):
                payload = Packet(data, address)
                ahh = False                         # if it is a recognized address
                for connection in self.connects:
                    if address == connection.address:
                        ahh = True
                        if string[1] == "DATA":         # handling of "DATA" packets
                            for number in connection.received:    # number list keeps track of data packets to be ack'd
                                if number == string[2]:
                                    buffer = buffer_builder("ADAT", number, window_size, "")
                                    self.sock.sendto(buffer, address)
                            if len(connection.buffers) < window_size:
                                i += 1
                                connection.received.append(string[2])
                                connection.buffers.append(payload)     # each connection gets its own queue of received packets
                        if string[1] == "RQCL":
                            connection.connected = False
                            self.connects.remove(connection)
                            packet = buffer_builder("ACLS", 1, window_size, "")
                            self.sock.sendto(packet, address)
                        if string[1] == "RCON":
                            # print("request received")
                            buffer = buffer_builder("AKRQ", 1, window_size, "")
                            self.sock.sendto(buffer, address)
                for request in self.requests:               # requests are connections that have not yet been accepted
                    if address == request.address:
                        ahh = True
                        if string[1] == "DATA":
                            if len(request.buffers) < window_size:
                                request.buffers.append(payload)
                        if string[1] == "RQCL":
                            self.requests.remove(connection)
                            packet = buffer_builder("ACLS", 1, window_size, "")
                            self.sock.sendto(packet, address)
                if ahh == False:
                    if string[1] == "RCON":
                        connection = Connection(address, False)
                        self.requests.append(connection)
                        buffer = buffer_builder("AKRQ", 1, window_size, "")
                        self.sock.sendto(buffer, address)
                    if string[1] == "RQCL":
                        # print("ahhh")
                        packet = buffer_builder("ACLS", 1, window_size, "")
                        self.sock.sendto(packet, address)
            lock.release()
            time.sleep(1)


def mrt_open(ip, port):
    global connection
    UDP_IP = ip
    UDP_PORT = port
    sock = socket(AF_INET, SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    connection = (ip, port)
    server = Server(sock)
    return server


def mrt_accept1(server):
    lock.acquire
    if len(server.requests) != 0:
        # print(1)
        connection = server.requests[0]             # moves connection from requests to connects
        server.requests.remove(connection)
        address = connection.address
        server.connects.append(connection)
        buffer = buffer_builder("ACON", 1, window_size, "")    # sends an ack
        server.sock.sendto(buffer, address)
        connection.connected = True
        return connection
    else:
        while len(server.requests) == 0:
            time.sleep(1)
        # print(2)
        connection = server.requests[0]
        server.requests.remove(connection)
        address = connection.address
        server.connects.append(connection)
        buffer = buffer_builder("ACON", 1, window_size, "")
        server.sock.sendto(buffer, address)
        connection.connected = True
        return connection
    lock.release


def mrt_accept_all(server):
    array = []
    lock.acquire
    for request in server.requests:
        server.connects.append(request)
        server.requests.remove(request)
        buffer = buffer_builder("ACON", 1, window_size, "")
        server.sock.sendto(buffer, request.address)
        request.connected = True
        array.append(request)
    lock.release
    return array


def mrt_connect(sock, connect):
    buffer = buffer_builder("RCON", 1, window_size, "")
    i = 1
    sock.settimeout(5.0)
    while i == 1:
        try:
            data = sock.recv(1000)
            string = buffer_reader(data)
            if string[1] == "AKRQ":     # checks for ack
                i += 1
                return string[3]
            if string[1] == "ACON":
                i += 1
                return string[3]
        except timeout:
            print("Connect timed out, resending...")
            sock.sendto(buffer, connect)


def hash_djb2(s):                   # generates checksum
    hash = 5381
    for x in s:
        hash = (( hash << 5) + hash) + ord(x)
    return hash & 0xFFFFFFFF


def buffer_builder(packet_type, sequence_number, window, data):         # My packet packer
    if data != "":
        string = packet_type + str(sequence_number) + str(window) + data
        checksum = hash_djb2(string)
        i = len(data)
        format_string = "l4sii%ds" % i                   # last variable in format string depends on length of data
        buffer = pack(format_string, checksum, packet_type.encode('utf-8'), sequence_number, window, data.encode('utf-8'))
    else:
        string = packet_type + str(sequence_number) + str(window)
        checksum = hash_djb2(string)
        buffer = pack("l4sii", checksum, packet_type.encode('utf-8'), sequence_number, window)
    return buffer


def buffer_reader(buffer):
    i = len(buffer) - calcsize("l4sii")             # calculating length of data
    if i != 0:
        try:
            format_string = "l4sii%ds" %i
            string = unpack(format_string, buffer)
            checksum = string[0]
            packet_type = string[1].decode('utf-8')
            sequence_number = string[2]
            window = string[3]
            data = string[4].decode('utf-8')
            result = [checksum, packet_type, sequence_number, window, data]         # returns a tuple
        except UnicodeDecodeError:
            result = [1000, "BADD", 1, window_size, ""]
    else:
        try:
            string = unpack("l4sii", buffer)
            checksum = string[0]
            packet_type = string[1].decode('utf-8')
            sequence_number = string[2]
            window = string[3]
            result = [checksum, packet_type, sequence_number, window]
        except UnicodeDecodeError:
            result = [1000, "BADD", 1, window_size, ""]
    return result


def mrt_send(sock, size, filename):
    file = open(filename, "r")
    data = file.read(980)
    i = 1                   # sequence number counter
    j = 0
    counter = -1            # ack counter
    buffers = []
    while data != "":       # takes data from a file and turns them into buffers
        buffer = buffer_builder("DATA", i, size, data)
        buffers.append(buffer)
        data = file.read(980)
        i += 1
    while len(buffers) != 0:
        print(len(buffers))         #remaining packets to send
        if counter != 0:
            j = 0
            while j < size:
                if j < len(buffers):
                    sock.sendto(buffers[j], connection)
                    j += 1
                    counter = 0
                else:
                    break
        else:                   # reads for acks
            counter += 1
            x = 0
            while x < size:        # determines how many times to check
                if x > j:
                    break
                try:
                    data = sock.recv(1000)
                    unpacked_ack = buffer_reader(data)
                    if unpacked_ack[1] == "ADAT":
                        x += 1
                        for buffer in buffers:
                            unpacked_buffer = buffer_reader(buffer)
                            if unpacked_ack[2] == unpacked_buffer[2]:
                                buffers.remove(buffer)
                            if unpacked_ack[1] == "ACLS":
                                sys.exit()
                except timeout:
                    break
    file.close()


def checksum_validate(unpacked_buffer):
    checksum = unpacked_buffer[0]
    if unpacked_buffer[1] == "BADD":
        return False
    if len(unpacked_buffer) == 5:
        string = unpacked_buffer[1] + str(unpacked_buffer[2]) + str(unpacked_buffer[3]) + unpacked_buffer[4]
    else:
        string = unpacked_buffer[1] + str(unpacked_buffer[2]) + str(unpacked_buffer[3])
    validate = hash_djb2(string)
    if checksum == validate:
        return True
    return False


def mrt_receive1(connection, server):
    i = 1
    lock.acquire
    while connection.connected == True:
        for number in connection.received:
            ack = buffer_builder("ADAT", number, window_size, "")
            # print("sent ack:" + str(number))
            server.sock.sendto(ack, connection.address)
            connection.received.remove(number)
        for packet in connection.buffers:
            string = buffer_reader(packet.packet)
            if string[1] == "DATA":
                if string[2] == i:
                    print(string[4])
                    connection.buffers.remove(packet)
                    i += 1

    lock.release
    time.sleep(1)


def mrt_probe(server):
    lock.acquire
    for connection in server.connects:
        if len(connection.buffers) > 0:
            lock.release
            return connection
        else:
            lock.release


def mrt_disconnect(sock):
    buffer = buffer_builder("RQCL", 1, window_size, "")
    while True:
        sock.sendto(buffer, connection)
        try:
            data = sock.recv(1000)
            unpacked_ack = buffer_reader(data)
            if unpacked_ack[1] == "ACLS":
                sys.exit()
        except timeout:
            continue


def mrt_close(server):
    buffer = buffer_builder("CLOS", 1, window_size, "")
    for connection in server.connects:
        server.sock.sendto(buffer, connection.address)
    for connection in server.requests:
        server.sock.sendto(buffer, connection.address)
    server.close = True
    sys.exit()

