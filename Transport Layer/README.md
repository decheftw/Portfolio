# README.md for Gao Chen's Computer Networking Lab 3
## Transport layer

### How to run
1. Sender
To test the sender I took a .txt file and ran `cat file.txt | python3 sender.py`
To test the receiver I ran `python3 receiver.py 1`

### Assumptions
1. I assume that the sender will not ctrl-c or force quit the process as the receiver
will not receive a disconnect request and it will continue to wait for packets.
2. I also assume the receiver will not force quit the process as the sender will
think the receiver is busy and keep sending/waiting for acks.


### Requirements
1. Testing against packet loss:
    a. I used Clumsy to test against packet loss, testing both inbound and
    outbound packets at up to a 50% drop rate. I checked that all needed
    data still ended up making it to my receiver. I had the sender print out
    acks that it received and the receiver print out packet numbers so that
    I could see what packets were being dropped and ensure that they were
    being resent.
2. Testing against data corruption:
    B. I used Clumsy and tested both inbound and outbound packets and redoing 
    checksum. In addition to ensuring that all required data eventually reached
    the receiver, I printed packet headers to ensure I could see which packets
    were being corrupted aside from the ones carrying data.
3. Testing against out of order delivery:
    a. Again used Clumsy at an up to 50% out of order rate. 
4. High latency:
    a. Used Clumsy to simulate high latency and ensured all necessary packets
    were received.
5. Testing with small amounts of data:
    a. I used an abridged version of Project Gutenberg's constitution. It is
    included in the github. I tested it with all 4 previous items.
6. Testing with large amounts of data:
    a. I used the full Constitution, also included in the github. Also tested
    with all 4 previous items. I also used a slightly shorter version so that it
    was more reasonable to check, instead of having to read the whole constitution
7. Flow control:
    a. I used two instances of wsl and fed the constitution through the sender
    while the receiver was running. The second instance did nothing while
    the first instance was interacting, and after the first instance disconnected
    the second instance started interacting.
8. Sender:
        a. mrt_connect
        b. mrt_send
        c mrt_disconnect
    Receiver:
        a. mrt_open
        b. mrt_accept1
        c. mrt_receive1
        d. mrt_close
    Not Used:
        a. mrt_accept_all
        b. mrt_probe
9. Testing not used:
    I wrote a version of the receiver where the connection that was chosen
    wasn't the most recent accepted connection, but one returned by mrt_probe.
    I wrote a third version of the receiver where mrt_accept_all was called
    and I ran the receiver with fewer than N senders.
10. All 9 functions (and some helper functions) can be found in mrt.py
###PROTOCOL
1. Connection handshake:
A receiver calls mrt_open.
A sender calls mrt_connect.
mrt_connect sends the target receiver a request to connect packet.
The mrt_open background process responds with an ack containing the window size.
If mrt_connect doesn't receive an ack, it sends another request to connect.

2. Sending: \n 
The sender reads from stdin and outputs to a file that it passes to mrt_send.
mrt_send draws the amount of bytes that it needs from the file and creates a 
list of buffers to send to the receiver. It sends the exact number of the 
receiver's window. If the receiver doesn't reply with acks, they could be busy,
but also all of my packets might have been dropped, or the receiver's acks might
have been dropped, so the sender resends all packets after a timeout if no acks
are received. 
As acks are received, mrt_send compares the sequence numbers on the acks to the
sequence numbers in the list of buffers to send. If there are matches, the packet
is removed from the list. This runs until there are no more buffers to send.
Packets that were sent but didn't receive an ack remain in the list to be sent
again. The last act of mrt_send is to delete the file that the sender created.
As the mrt_receive1 receives packets, it keeps a tracker starting at 1. It
continuously goes through all received packets for the active connection
until it finds the one with the matching sequence number, ensuring that they
print in order. However, acks are sent as soon as packets are received.

3. Disconnecting: 
When the sender finishes sending packets, it calls mrt_disconnect, sending a 
disconnect message to the server. If the server receives the disconnect, it sends
an ack and moves on to the next sender. If it doesn't, it continues to wait for
packets. mrt_disconnect will resend the disconnect request.
When the receiver receives a disconnect request, it accepts a new connection.
It already has an amount of data packets from that connection equal to the window
size, so it prints those before moving on to new received packets.
The receiver will wait for senders until it has reached N senders.

