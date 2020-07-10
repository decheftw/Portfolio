# Lab 3: Mini Reliable Transport Layer

Shikhar Sinha. Professor Joosten. Computer Networks, Spring 2020

Find a description of my protocol [here](./PROTOCOL.md), including
packet structures and how I meet requirements 1-7 of the assignment
spec.

### Running Sender and Receiver
The _sender_ and _receiver_ programs are found in
[sender.py](./sender.py) and [receiver.py](./receiver.py) respectively.
Run the _receiver_ with (on Ubuntu 18.04) _python3 receiver.py n_ where
_n_ is the number connections you wish to accept. Run the _sender_
with _python3 sender.py_ and it will send the first thing you enter to
stdin. However, it can be more useful if you pipe output from another
file into the function. For example: _cat ./README.txt | python3 sender.py_
will take the contents of the README, initiate a connection to a server,
and send the contents of the file over the MRT Layer.

### Overview of Project/Code Organization
Find my code for the relevant functions in the [mrt.py](./mrt.py).
The code for the functions is broken down into several classes
to divide functions appropriately and aide user experience.

#### Server
The _Server_ Class is the endpoint through which receivers are
created. 
1. Call `mrt_open(port)` with a port on which you wish to listen.
  If that port is available, the server will listen
  at that port. If a socket cannot be bound there, `mrt_open()`
  will try to bind at the default port (22222). If none of these ports
  are available, the function call will return _None_. However,
  if the function is able to bind to either port, it will return
  a _Server_ object and initiate a thread to handle incoming
  packets through that port
2. Call `mrt_accept1()` or `mrt_accept_all()` on the server
  where you wish to accept a connection. i.e.
  __someServer.mrt_accept1()__. _accept1()_ will block until it
  eventually returns a ServerConnection object to handle
  incoming data for the accepted connection. __accept_all()__
  will return a possibly empty list of connections that are
  ready to be excepted

3. Call `mrt_close()` on a _Server_ object to stop that
  object from handling any more data and close the socket

#### ServerConnection
The _ServerConnection_ Class provides an object through which
a user can access data to a particular connection. Obtain a
_ServerConnection_ object from `mrt_accept1()` or `mrt_accept_all()`
calls on a given _Server_ object.

1. Call `mrt_receive1()` on a _ServerConnection_ object to block
  and until bytes are available and read those bytes. If the
  sender disconnects, return _None_

2. Call `mrt_probe()`, a static function, passing in a list of
  _ServerConnection_ objects to return one, if any, 
  _Server_Connection_ object with available data. If no such
  object is available, return _None_

#### SenderConnection
The _SenderConnection_ Class provides an object through which
a user can send data to a receiver.

1. Call `mrt_connect(dest_ip, dest_port)` (static function) with
  the ip address and port of your target server/receiver. This
  will return a _SenderConnection_ object. It will block until
  the connection is available, and will print a message if the
  target server/receiver is closed

2. Call `mrt_send(data)` on a _ServerConnection_ object with 
  the data you wish to send from a connected _ServerConnection_
  object to a server/receiver. This function will block until
  all the data you attached is sent and acknowledged, or until it
  receives a message indicating the server/receiver is closed.

3. Call `mrt_disconnect()` on a _ServerConnection_ object to
  disconnect it from a server/receiver. This function will
  send a Disconnect message until it receives an acknowledgement
  of the disconnect, the server sends a message indicating closure
  or a set period of time has passed (currently set to 15 seconds)

## Testing Key Features
For many of these features, I used the *_clumsy_* tool linked
in the course wiki.

For simplicity, unless otherwise stated, 
I will describe results from running my receiver
with N = 1 (_python3 receiver.py_), and the sender sending data
from my lab1 README (_cat ../lab1/README.md | python3 sender.py_).


### 1. Testing against packet loss
I used the clumsy tool, setting the drop rate to 50%.
I noticed a slowdown in overall transfer rate, but I was still
able to continue transferring data

### 2. Testing against data corruption

To test against data corruption, I used the clumsy tool. I ticked
the _Tamper_ flag, set the "Redo Checksum" flag to prevent the
UDP layer from filtering this data out, and set the Chance value
to 100%. Initially, I see no connection being made or any output
on the receiver side. With some debug print statements deployed
on both the sender and receiver, I can see that the sender is
repeatedly sending syn messages and that the receiver is
consistently dropping packets. On the sender connection object,
I implemented a print statement to indicate when a syn message
was being sent, and that was the only output of the run. Syn
messages were sent repeatedly but nothing was happening. All
packets were being filtered out by the checksum verification.
To verify that ack_packets were protected against data corruption,
I manually changed the checksum in packets from receiver to sender.
This way the checksum was guaranteed to be wrong. On the sender side,
the connection object ignores corrupted packets and prints a message
indicating the checksum did not match. I saw this occur, and no connection
was ever established. To see this in action, uncomment the second to last line
in the __build_receiver_packet()__ function in mrt.py.

### 3. Testing against out-of-order delivery

To test against out-of-order deliver, I once again used the clumsy tool.
I set the "Out of Order" flag and set the "Chance to 100 %." At this point
I was still able to see successful transmission. Further, I added a
print statement into my __handle_packet()__ function that handles a given
data packet. For each data packet received in the window, I printed
the sequences numbers as they arrived. I confirmed that the packet
sequence numbers were not in ascending order, and also confirmed that the
data was being reassembled correctly. The print statement is commented out
in my code but contains the phrase 
received sequence number " + str(packet.sequence_number)" if you would like
to turn on clumsy and see for yourself.

### 4. Testing against high-latency delivery

To test for high latency delivery, I artifically set the inital window and
packet conditions to produce an increase in transmiision rate. The
default values for Maximum Data in Outbuffer, Initial Sender Window,
and Initial Server Window are all 256 bytes. The packet size is set to 64
bytes. These values are listed in the mrt.py file at the end of the list
of declared constants, at approximately line 40-45. I changed the values
to exaggerate the appearance of taking advantage of high latency delivery.
I set MAX_OUTBUFFER and RECEIVER_WINDOW to 1024 to prevent these from being
limiting factors, and I set the INIT_SENDER_WINDOW to 8. The PACKET_SIZE was
also set to 8 bytes. This time, when I ran it under the situation as
described at the top of this section, I also used some print statements in
the `mrt_send()` function that printed the number of bytes sent through
each window, as well as a print statement whenever the window size increased.
These print statements are commented out but read
"print("Sent all bytes in window: " + str(bytes_sent_in_window))" and
"print("Increasing window size")." Uncomment these and you should see the same
thing I saw: the bytes sent in window print statement consistently increased by
8 until the end of the file, eventually becoming a 224-byte sender window 
before being cut off by the end of the file. Between each of these, there was
a message indicating an increase in window size. 

### 5. Testing sending small amounts of data

To test for small amounts of data, I set the Packet size to be the size of
the window (256 bytes) and sent the output of _pwd_ to my receiver. I set 
the same print statement as in the above section to determine how many
bytes of data were sent. My path at the time was:
"/home/shikhar/networks_shikhar/lab3." My sender showed 36 bytes being sent,
less than the size of one packet.
### 6. Testing with large amounts of data

To make sure that large amounts of data worked well and didn't all end up
with the receiver too quickly, I used N = 2, and first connected and sent a
long file (my README from lab2), and then waited a minute and connected
with another sender trying to send the result of "pwd" (36 bytes). I also
added a wait between consecutive receive1() calls and saw that not all data
for the larger data transfer was available at once.

### 7. Testing flow-control

Here, I used the same setup as above and the same print statement as in
section 5. With no data being moved to the outbuffer, the receiver
window and outbuffer fill up. On the sender side, I saw the sender send
fewer and fewer bytes until the receiver was able to free up space
by releasing and printing data from the outbuffer


### 8. Describe which functions your 'sender' and 'receiver' program did and did not use
My sender used `mrt_connect()` to establish connections, `mrt_send()` to
transfer data, and `mrt_disconnect()` to end the connection.

My receiver used `mrt_open()` to open a server. It called `mrt_accept1()`
N times to make N connections. Then for each connection it called
`mrt_receive1()` to print data. Finally, it called `mrt_close()` to close
the original server.

I did not use `mrt_probe()` or `mrt_accept_all()` in these versions of the
sender and receiver

### 9. Describe how you tested the functions that weren't used in 'sender' or 'receiver'

To test `mrt_probe()`, I reconfigured my receiver to continually probe a set consisting
of one sender (the connection of interest) until it disconnected, printing whatever data
was available. This code is commented out in the submitted version of receiver. To see
it in action, comment out the current comments of the while loop and uncomment what is
currently commented out just below it.

I did a similar thing to test `mrt_accept_all()`. I had the server open up, then I
put the sender to sleep for 15 seconds. After, I started 3 senders and called
accept_all() once instead of calling accept1() 3 times. Then, ran the
remainder of receiver as normal. To see this in action, comment out the code
within the for loop and instead uncomment the two lines above it


### 10. Where to find the code

All the relevant functions are in [mrt.py](./mrt.py). open(),
accept1(), accept_all() and close() are in the Server Class.
receive1() and probe() are in the ServerConnection Class, and
connect(), send(), and disconnect() are in the SenderConnection
Class.