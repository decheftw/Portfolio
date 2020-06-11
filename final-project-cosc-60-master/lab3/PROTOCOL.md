# Mini Reliable Transport Protocol

This protocol and implementation are designed to provide one-way reliable data
transfer over UDP. It is adapted from TCP and is stream oriented. This
document provides an explanation of how the protocol works and how certain
features are guaranteed. See the packet structures at end of document.

### Setup
Data is transferred from a sender to a receiver. First, we set up the server
on the receiving end to listen for new connections. Then we initialize the 
sender to start the connection and data transfer process. As we'll soon see,
it is not strictly necessary for the server to be running first, but it will
aid in our explanation.

### Handshake
When `mrt_open` is called, the server starts listening on a provided 
UDP port (or a default port if one isn't available). If neither port is
available or the sokcet bind fails, then the server returns nothing
and announces a failure. Now, assuming the server is able to open the
socket, we turn to the sender end. The sender calls `mrt_connect`.
This opens a socket to send data to the server to initiate a connection.
At the same time, the sender chooses a random inital number (between 0
and 4095) for the initial sequence number, and sets its initial sender
window. By starting our sequence number between 0 and 4095 and using
a 64-bit sequence number, we don't have to deal with sequence
number overflow unless a sender decides to send over 16 exabytes of data.

At this point, the connection establishment behaves like TCP. Once the
socket is open, the sender sends a syn message communicating
its initial sequence number. The server sees the syn packet and responds
indicating receipt of the pack and the sequence number by responding
with a syn-ack containing an acknowledgement number that is one more
than the original sequence number. This is now the sequence number for
the remainder of the connection. Upon receipt of the syn-ack, the
sender sends an ack back to the server. At this point, the sender
considers itself in the "connected/established" stage. It returns from
the `mrt_open` function call, passing back a _SenderConnection_
object through which the Application Layer will access mrt functions.
The server will also move this connection to the "connected/established" 
stage when it receives the ack message and return a _ServerConnection_
object to access functions associated with this connection.

#### Handling Packet Loss in the Handshake

1. The initial SYN gets lost. After sending the SYN message, the sender
  listens on the socket for a reply (the SYN-ACK). If the recv function
  times out, the sender must assume the SYN or SYN-ACK, so it sends it
  again. The sender will continue to do so until it succesfully receives
  a properly formatted SYN-ACK. Thus, it is ok for the sender to call
  `mrt_open` before the server makes itself available because it will
  continually send SYN messages.
2. The SYN-ACK gets lost. At this point, the sender has already sent a
  SYN and is waiting for the SYN-ACK. Therefore, the sender receive will
  timeout, and not knowing whether the SYN did not arrive or the SYN-ACK
  got lost, must send the SYN again. When the sender receives a SYN from
  the same address (host/ip, port) it has already seen, it resends a
  SYN-ACK because the SYN-ACK it first sent must have gotten lost. It will
  continue to do so for SYNs from this address until this connection
  moves to the connected/ready state.
3. The ACK gets lost. At this point, the sender is already in the
  established state, so it may start sending data. The server is listening
  for an ACK, but may actually receive a data packet instead. In this case,
  the server treats this data packet as an implict ACK and moves the
  connection to the ready/established state. Further, it sends an ACK
  message indicating the sequence number

### Accepting a Connection

The sender is already ready to send data and may do so, but the server
cannot handle data for a connection that hasn't yet been accepted, so
any packets it receives for a connection that has not yet been accepted
are dropped. From the handshake however, we know both sender and server
are ready (because the first data packet the server received put this
potential connection in the ready state). When `mrt_accept1()` is called,
we will repeatedly check for an available connection in the ready queue
storing such connections. When one is available, we remove it from the
ready queue, and add it to our dictionary of established connections.
This means subsequent data packets for this connection will be handled.
`mrt_accept1()` will return a _ServerConnection_ object for performing
further actions with this connection. As currently configured, the
connection object maintains a window buffer for incoming data that isn't
ready to be added to the connection's outbuffer from which `mrt_receive1()`
pulls data. The outbuffer if of limited size so if the outbuffer and window
fill up, subsequent data won't be accepted. The receiver connection can hold
data up to the size of its outbuffer plus window size after the connection
is accepted and before `mrt_receive1()` is called on it.

### Data Transfer

Data transfer from the _SenderConnection_ object to the _ServerConnection_
can occur as soon as the `mrt_connect()` call returns, though there will
be a limit on the total data transfer until the connection is accepted, 
and after that, when sufficient data is not yet read from the outbuffer
by `mrt_receive1()`.

Let's examine the data transfer aspect of this MRT protocol from when
`mrt_send()` is called until when it returns. The send call won't return
until it has received an ack corresponding to the last byte it intends to
send. The sender has an intrinsic packet data size that it uses to chunk
inputted data. If the entire data or the last chunk is not equal to one
complete packet, this packet will be correspondingly smaller (there is 
no padding of data). The sender, from the connection
establishment process, is aware of the receiver's window size. It also
maintains its own sender window size. The sender takes the minimum of these
two values and uses this as its effective window (much like TCP). The size
of this effective window is the maximum amount of unack'd bytes that the
sender allows to be outstanding. The sender will never send more bytes out
at a time than the size of its effective window. Note, the size of the
sender window is dynamic and changes in response to how the previously sent
packets were received.

* Until all bytes have been ack'd:
  * Determine effective window, and until you have sent this many
    bytes or sent all the desired bytes, do the following:
    * Take the first packet's worth of data (offset from the beginning
      of provided data by the number of bytes ack'd thus far) and
      send it to the receiver.
    * Continue chunking and sending the data until you have sent all the
      desired data
    * Once you have sent all the data, listen for acks. For each ack you
      receive, check the acknowledgement number. Keep track of the highest
      acknowledge number you see. Also, check the packet and update your
      record of the receiver's window Do this until the listener for acks
      times out.
    * At this point, look at the maximum ack_number. The difference between
      this value and the original sequence number before sending out this
      window's worth of bytes is the number of bytes ack'd. Up your counter
      of this and update your sequence number to this highest ack number.
      On the subsequent window, start from this new offset. Here, we can
      perform some congestion control and flow control. If no ACK messages
      were received, or ACKs were only for our original sequence number,
      we should reduce our send rate. In this case, we divide our sender
      window and truncate to the nearest packet-size multiple. On the other
      hand, if the receiver sends acks back for all our data, we can
      increase our transfer rate. We do so by increasing our sender window
      by one packet's worth of data.
    * Start sending with the new window. Repeat this process until all the
      data has been successfully transferred and ACK'd
 
 * Now, let's look at the sending process from the receiver connection side.
   Once the connection is in the ready state, we can handle data packets as
   they come in. For each packet, we check if it fits entirely in our
   window. If it doesn't, we drop the packet. If the packet has no data,
   we don't do anything with this packet but send an ACK for our current
   sequence number. Now, we know the packet fits in our window. We store
   the packet in our window offset by its position in the current window.
   At this point there are two possibilities

   1. If this packet corresponds to the beginning of the window, we look
     at out window and pull all the consecutive bytes of data that we
     may have stored previously starting at the beginning of the window.
     If there is room in our outbuffer for this much data, append it to
     the outbuffer. We reset our window accordingly and update our
     sequence number accordingly. Send an ack for this new sequence
     number to proceed with data transmission. If there is room
     for some of the data but not all, then we move what we can and 
     update our window and sequence number accordingly. Send an ack for
     this new sequence number. On the other hand, if there  is no room
     we don't do anything with the packet and just store it in the window
     and ask for packets to fill up the window.
   2. This packet falls somewhere in the middle of our window. At this point
     We store in a dictionary of ranges a mapping from the first sequence
     number of this packet to the last sequence number of this packet. This
     way, when we receive a packet corresponding to the beginning of the
     window, we can pull al consecutive data we have seen so far. We don't
     usually send an ACK for such data because we want to account for
     out-of-order delivery. However, if it has been too long since a packet
     at the front of the window arrived, we will send a corresponding ack.

### Disconnect and Close

* `mrt_disconnect()`: We use a 2-way disconnect process. When
  `mrt_disconnect()` is called, the sender will send a CLS message and wait
  to hear either a ACK-CLS message, or a FIN message indicating the that
  the server is closed. Because we don't want the sender to try to close
  indefinitely (as might happen with a closed receiver), the server
  will try to close kindly for a set period of time at which point
  it will perform a hard close.
* `mrt_close()`: On a close() call, the server will send one message to
  all connections (open and pending) with a FIN flag indicating that it is
  no longer accepting data. It will not send more than once and once it has
  completed sending, it will exit.

## How I Support Requirements 1-7 in the Assignment Spec

### 1. Protection against packets dropped on the UDP layer

As discussed in the data transfer section, I require ACKs to move
forward with sending additional data. This means that until the sender
gets confirmation for a particular byte, it will send it again and again.
In this manner, data in packets that are droppped will eventually make it
as long as the underlying network is not completely unreliable.

### 2. Protection against data corruption

In the construction of each packet, I implement a 32-bit Cyclic
Redundancy Check (CRC) that is calculated on the remainder of the packet.
Any packet that is sent contains a checksum, and the first step in
processing a received packet is to verify the checksum. I uses
Python's zlib library, and its _crc32()_ function.

### 3. Protection Against Out-of-Order Delivery

As discussed in the data transfer section, I store packets in a window
and keep track of the ranges of sequence numbers of packets that have
already arrived. This way, if a packet arrives that isn't the first
packet I am expecting, I can store it until consecutive data up through
those bytes arrives. Further, if packets arrrive out of order, I can
store them until the first packet in the window arrives. Then, when
there are consecutive packets, I can pull the consecutively stored
packets and put them in the outbuffer. However, there is a time limit.
If an out-of-order packet arrives too long after the last time data
was available at beginning of the window, we can no longer assume that
packets are coming in order, so we issue an ACK at this point.

### 4. Fast Transmission if Latency is High

As discussed in the data transfer section, if an entire window was sent
successfully, I will increase my window size by one packet for
subsequent data transfers. This way I can increase transfer rate if the
connection is sufficient. In conjunction with my implementation of
step 7, flow, control, I follow TCPs Additive-Increase/Multiplicative
Decrease feedback system.

### 5. Transmission of Small Amounts of Data

The packets are as small as the data that's sent in them. So if a sender
calls send with 2 bytes of data, the packet will be 2 bytes long plus
the the length of the header (15 bytes).

### 6. Transmission of Data that does not fit in Memory

As discussed in the data transfer section, the receiver has a maximum
quantity of unread data (the size of the window and the outbuffer)
after which subsquent data won't be accepted/acknowledged. This way,
the receiver does not have a boundless buffer that will grow and
grow. This way, there is a cap on the total outstanding data. It is
limited by the window and outbuffer on the receiver side and also
by the sender window on the sender side.

### 7. Flow Control for Congestion at the Receiving End

As discussed in the data tranfer section, the sender will decrease
the size of its window if the receiver did not ack any of the data
that was just sent. The sender will shrink its window, repeatedly
dividing by 2 all the way down to the a limit of one packet's worth
of data

## Packet Structures:

### Sender to Receiver:

___________________________________________

Checksum 32 bits

Sequence Number64 bits

Flags (Syn,Ack,CLS,FIN), 4 bit Padding (0)

Data Length 16 bits

Data


### Receiver to Sender
___________________________________________


Checksum 32 bits

Acknowledgement Number bits 64 bits

Flags (SYN,Ack,CLS,FIN),4 bit Padding(0)

Data Length 16 bits

