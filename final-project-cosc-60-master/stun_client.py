'''
Shikhar Sinha
Professor Joosten
Computer Networks Lab 4
Spring 2020

STUN client for NAT Behavior Discovery

'''

import random
import socket
import time

MAGIC_COOKIE = 0x2112A442
STUN_ADDR = ('stun.voipstunt.com', 3478)

# Check that a given response from the STUN serve has the appropriate transaction id
def verify_response(tid, resp):
    if resp[0] != 1:
        if resp[0] == 0 and resp[1] == 1:
            print("An error response!! HANDLE THIS")
            return False
        print("Incorrect message type")
        return False
    if resp[1] != 1:
        print("Incorrect message type")
        return False
    if (resp[2] << 8 ) | resp[3] != len(resp) - 20:
        print("Error with incorrect message length matching")
        return False
    if (resp[4] << 24) | (resp[5] << 16) | (resp[6] << 8) | resp[7] != MAGIC_COOKIE:
        print("Magic Cookie doesn't match")
        return False
    if ((resp[8] << 88) | (resp[9] << 80) | (resp[10] << 72) | (resp[11] << 64) | (resp[12] << 56) | (resp[13] << 48)
        | (resp[14] << 40) | (resp[15] << 32) | (resp[16] << 24) | (resp[17] << 16) | (resp[18] << 8) | resp[19]) != tid:
        print("Error: Transaction ID does not match")
        return False
    return True

# Compute the tid of a request
def get_tid(request):
    return ((request[8] << 88) | (request[9] << 80) | (request[10] << 72) | (request[11] << 64) | (request[12] << 56) | (request[13] << 48)
    | (request[14] << 40) | (request[15] << 32) | (request[16] << 24) | (request[17] << 16) | (request[18] << 8) | request[19])

# Build a STUN request
def build_request(attributes):
    stun_msg_type_1 = 0
    stun_msg_type_0 = 1

    msg_len_1 = 0
    msg_len_0 = 0

    if len(attributes) != 0:
        attr_len = 0
        for attribute in attributes:
            attr_len += len(attribute)
        msg_len_1 = attr_len >> 8
        msg_len_0 = attr_len & 0x00FF

    cookie_3 = MAGIC_COOKIE >> 24
    cookie_2 = (MAGIC_COOKIE >> 16) & 0x00FF
    cookie_1 = (MAGIC_COOKIE >> 8) & 0x0000FF
    cookie_0 = MAGIC_COOKIE & 0x000000FF

    transaction_id_11 = random.randint(0,255)
    transaction_id_10 = random.randint(0,255)
    transaction_id_9 = random.randint(0,255)
    transaction_id_8 = random.randint(0,255)
    transaction_id_7 = random.randint(0,255)
    transaction_id_6 = random.randint(0,255)
    transaction_id_5 = random.randint(0,255)
    transaction_id_4 = random.randint(0,255)
    transaction_id_3 = random.randint(0,255)
    transaction_id_2 = random.randint(0,255)
    transaction_id_1 = random.randint(0,255)
    transaction_id_0 = random.randint(0,255)
    packet = [stun_msg_type_1, stun_msg_type_0, msg_len_1, msg_len_0, cookie_3, cookie_2, cookie_1,cookie_0,
                        transaction_id_11, transaction_id_10, transaction_id_9, transaction_id_8, transaction_id_7, transaction_id_6,
                        transaction_id_5, transaction_id_4, transaction_id_3, transaction_id_2, transaction_id_1, transaction_id_0]
    if len(attributes) != 0:
        for attribute in attributes:
            packet.extend(attribute)
    return bytearray(packet)

def client(sock):
    request = build_request([])

    sock.sendto(request, STUN_ADDR)
    new_timeout = 0.5
    sock.settimeout(new_timeout)
    resp = None
    timeouts = 0
    while timeouts < 7:
        try:
            resp = sock.recv(0x10000)
        except socket.timeout:
            new_timeout *= 2
            sock.settimeout(new_timeout)
            timeouts += 1
            sock.sendto(request, STUN_ADDR)
        if resp != None:
                break
            
    if resp == None:
        print("Unable to Reach STUN Server")
        return None


    tid = get_tid(request)
    if not verify_response(tid, resp):
        print("Error with the packet")
        return None
    

    init_response = handle_packet(resp)

    if init_response == None:
        print("unable to contact properly")
        return None

    print("According to the STUN Server, I am " + init_response[0] + ":" + str(init_response[1]))

    return (init_response[0], init_response[1])


    
# Handle the info contained in a packet
def handle_packet(resp):
    pos = 20
    own_ip = None
    own_port = None
    src_ip = None
    src_port = None
    alt_ip = None
    alt_port = None
    while pos < len(resp):
        attr_type = (resp[pos] << 8) + (resp[pos+1])
        attr_len = (resp[pos+2] << 8) + (resp[pos+3])
        pos += 4
        # print("attr_len: " + str(attr_len))
        # print("attr_type: " + str(attr_type))
        if attr_type == 0x0001: #MAPPED ADDRESS
            # print("MAPPED_ADDRESS")
            if resp[pos] != 0:
                print("Error With 1st Byte")
                return
            own_port = ((resp[pos + 2] << 8) + resp[pos + 3])
            # print("Own Port: " + str(own_port))
            if resp[pos + 1] == 0x01: # IPv4
                # print("IPv4 Address")
                own_ip = (str((resp[pos+4])) + "."
                         + str((resp[pos+5])) + "."
                         + str((resp[pos+6])) + "." 
                         + str((resp[pos+7])))

            elif resp[pos + 1] == 0x02: # IPv6
                print("IPv6 Addresss")
            # print("own_ip_addr: " + str(own_ip))
                return

        elif attr_type == 0x0009: # ERROR CODE
            print("ERROR_CODE")
            return

        elif attr_type == 0x0020: # XOR_MAPPED_ADDRESS
            # print("XOR_MAPPED_ADDRESS")
            if resp[pos] != 0:
                print("Error With 1st Byte")
                return
            own_port = (((resp[pos + 2] << 8) + resp[pos + 3]) ^ (MAGIC_COOKIE >> 16))
            # print("Own Port: " + str(own_port))
            if resp[pos + 1] == 0x01: # IPv4
                # print("IPv4 Address")
                own_ip = (str((resp[pos+4]) ^ (MAGIC_COOKIE >> 24)) + "."
                         + str((resp[pos+5]) ^ ((MAGIC_COOKIE >> 16) & 0x00FF)) + "."
                         + str((resp[pos+6]) ^ ((MAGIC_COOKIE >> 8) & 0x0000FF)) + "." 
                         + str((resp[pos+7]) ^ (MAGIC_COOKIE & 0x000000FF)))

            elif resp[pos + 1] == 0x02: # IPv6
                print("IPv6 Addresss") 

            # print("own_ip: " + str(own_ip))

        elif attr_type == 0x0004: # Source Address
            # print("Source Address")
            if resp[pos] != 0:
                print("Error With 1st Byte")
                return
            src_port = (((resp[pos + 2] << 8) + resp[pos + 3]))
            # print("Src Port: " + str(src_port))
            if resp[pos + 1] == 0x01: # IPv4
                src_ip = (str((resp[pos+4])) + "."
                         + str((resp[pos+5])) + "."
                         + str((resp[pos+6])) + "." 
                         + str((resp[pos+7])))

            elif resp[pos + 1] == 0x02: # IPv6
                print("IPv6 Addresss") 

            # print("source_addr: " + str(src_ip))
        elif attr_type == 0x0005: # Changed Address
            # print("Changed Address")
            if resp[pos] != 0:
                print("Error With 1st Byte")
                return
            alt_port = (((resp[pos + 2] << 8) + resp[pos + 3]))
            # print("Alt Port: " + str(alt_port))
            if resp[pos + 1] == 0x01: # IPv4
                alt_ip = (str((resp[pos+4])) + "."
                         + str((resp[pos+5])) + "."
                         + str((resp[pos+6])) + "." 
                         + str((resp[pos+7])))

            elif resp[pos + 1] == 0x02: # IPv6
                print("IPv6 Addresss") 

            # print("change address: " + str(alt_ip))
        else:
            print("STUN Server did not respond as anticipated")
            # if attr_type == 0x0006: #USERNAME
            #     # print("USERNAME")
            #     return
            # elif attr_type ==  0x0008: # MESSAGE_INTEGRITY
            #     # print("MESSAGE_INTEGRITY")
            #     return
            # elif attr_type == 0x000A: # UNKNOWN-ATTRIBUTES
            #     # print("UNKNOWN_ATTRIBUTES")
            #     return
            # elif attr_type == 0x0014: # REALM
            #     # print("REALM")
            #     return
            # elif attr_type == 0x0015: # NONCE
            #     # print("NONCE")
            #     return
            # elif attr_type == 0x802C:
            #     # print("OTHER_ADDRESS")
            #     return
            # # print("UNSUPPORTED ATTRIBUTE TYPE")
            # print(attr_type)
            # # return
        pos += attr_len
        # print()
    return (own_ip, own_port, src_ip, src_port, alt_ip, alt_port)


if __name__ == '__main__':
    sock = client()
    # if sock != None:
    #     sock.sendto(bytes([1,2,3,4]), ('73.238.8.136', 59175)) # Sucharita's port and IP from testing
    #     print("receiving a message")
    #     data = sock.recv(0x10000)
    #     print(data)








