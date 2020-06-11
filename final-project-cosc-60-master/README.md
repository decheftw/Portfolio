# Zonnebloem: P2P File Sharing Client

This repository contains the source code for a P2P file-sharing protocol. This protocol is decentralized, and it can be used to create networks of at least one node.

To use these file-sharing features, useres will need to run our `peer_client.py`. Instructions for how to run this script and an overview of its functionality are included in the sections below. 

## Client

### How To Run This Code

- To join P2P file-sharing networks, you will need to host a node. To do this, run:

- python3 peer_client.py [PORT] [I/O DIRECTORY] [--public]

- requires a python 3 installation and that you have everything listed in
requirements.txt (just a pystun installation). This can be installed with pip
or another python package manager

This script takes two arguments:

- `PORT` — The port that you want your client to run on. This should be an integer.
 
- `I/O DIRECTORY` —  The directory that you want to add files to and send files from. 
This should be a filepath to a directory on your local machine. 
All requested files will be added to this directory.
All files that you wish to send must be contained in this directory.

- `--public` is an optional flag that sets the user's external IP equal to their private IP. This defaults to `True`.
Use this when testing between nodes that are behind different NATs. If testing where are all nodes are behind
the same NAT or are on the same machine, leave this flag unset.
### Interacting With The Client

Once you run the script, a built-in STUN client will fetch your external IPV4 address. Here's an example: `10.230.128.211:9001`. This is my `internal address`. 
I will send this to other users who have the `peer_client.py` source code, so they can connect to my machine. 

Next, you will be prompted to enter a username. It can be any valid alphanumeric ASCII string. It cannot contain spaces.

#### Once you have entered a username, you will be prompted to enter one of the following commands:

CONNECT [ip] [port]
- Connect to the given ip/port combination

ADD [file]
- Add the given file to the network. Must be in the given directory.

GET [file]
- Get the named file from the network.

LIST
- List the names of available files in the network.

IP
- Get the external IP Address of your node.

CHAT [text]
- Send the text to other users on the network.

HELP
- Display this help message again.

QUIT
- Exit the program.

## Protocol and Structural Overview

The two additional features we added were:

Efficient routes for exchanging files (as measured by fewest hops)

Facilitating chat between users, using a similar mechanism as the
broadcast functionality that will be discussed below

You can find the code for all the following features in [node.py](./node.py)


### Connection and Structural Logic

A given node is operating at a user-provided port.

On startup, the node will bind a UDP socket to that port and use a STUN
client implementation to determine its external ip and port. Additionally,
we start a mrt server to make and accept connections at this port Note, if a
user is behind a symmetric NAT, this information does not get us much. In
addition, the user will also determine its ip address behind the NAT so
that even if it is behind a Symmetric NAT, it will be able to share files
with nodes behind the same NAT.

Once the node announces both its external and local ip addresses, another
user can connect by inputting first the external client and port, and if
it turns out the user is behind the same NAT, the user will also be prompted
to enter the local ip and port targets, and we will use those.

To connect to this node, we rely on the mrt layer from Lab 3. However, we
made extensive modifications to allow for communication across NATs. The mrt
as originally assigned, provided one-way data transfer using a socket on both
sides. To perform the 2-way communication required for P2P file transfer, the
first thought might be to use 2 separate mrt connections with a total of 4
sockets. However, in a situation where one node is behind a symmetric NAT,
this will introduce significant complexity. It will be easy to make a one-way
mrt connection allowing for data transfer from the symmetric to permissive
side, but making the connection back in the other direction with the original
mrt format would be difficult because the socket receiving this communication
on the symmetric side has an unknown external port that will be unique to
communication with the sender on the permissive side. This would require a
complex information transfer and hole-punching procedure.

Instead, I modified the old mrt layer to provide 2-way data transfer using
one socket on each side while still providing the same abstraction of using
a separate object for communication in each direction. This allows for
commuinication where one node is behind a symmetric NAT while the other is
behind a Full Cone NAT. However, we were unable to test this modification
because nobody in our group had the symmetric NAT and full cone NAT required
to see this in action.

#### Message format
All messages are formatted using JSON for serialization as follows:

Each message is a 2-element JSON array where the first element is a string
indicating message type and the second element is a JSON object with the
corresponding keys and values.

Once the JSON is serialized into a string, we calculate the length of the
string, prepend that to the beginning of the JSON array and send that.
The message length facilitates deserialization an allows the receiver to
transform the byte stream provided by mrt into discrete messages.

#### Connection
Using the mrt server, a node A will connect to node B using the provided
user ip and port. Node B is using mrt_accept_all and mrt_accept_1 to
collect incoming connections. When B receives this connection, it will
also listen for a message indicating the external port and ip to associate
with this node and upon which it should index. Once it has this information,
it will use mrt_connect to create the corresponding data transfer connection
in the opposite direction and send over its external ip and port info so that
all nodes in the network refer to every other node in a consistent manner.
Once this data transfer is complete, the 2 nodes are connected. Then, each
node shares with this new connection its own file directory of discovered
files using the same broadcast functionality that will be discussed in the
following sections. 

### Testing
We are all behind restrictive NATs, so we had difficulty testing across the
open internet, but we were able to transfer files to each other when we
all connected to the Dartmouth VPN

### Connection Functions

1. start_node(): takes in a port number and directory. It binds a UDP socket
on that port and uses STUN to find the port and IP, then we determine our
IP address behind the NAT. After checking that the supplied directory is
writable, we create a node object, initalizing the necessary data structures
and start the packet handler thread. The packet handler thread runs in the
background and complete connections as necessary.

2. connect(): this takes in an ip and port and if this ip is the same as the
nodes, they are on the same computer or behind the same NAT, so we prompt for
a local ip and port to guarantee connection and user intent. Then, we initiate
a one-way connection to said ip and port (possible the local one as necessary)
Once connected, we send a message indicating our external ip and port, so that
all neighbor nodes refer to us in the same way, reducing ambiguity. The
connection is not complete until the neighbor node creates a connection back to
us. Connect returns once that connection is made and the node is ready to
function. This return connection is made in the packet handler thread

3. packet_handler(): This thread listens on the server for new connection. As
each connection comes in, it checks if we already have a connection object for
this address, indicating that we initiated this connection, or if this
connection is new to us. If we initiated this connection, we associate this
connection with the appropriate neighbor and mark that neighbor as ready.
Else, a node is trying to connect, so we read in the address message associated
with this connection, store this information, launch a connection for one-way
data transfer back in the other direction, upon connection completion, send over
our external address info, and mark this address as connected. In either case,
after this is over, for every file in our directory, we broadcast so that
new neighbors are aware of files in the network

### Utility Functions

1. receive_message_2(): uses JSON deserializing to transform the bytestream
provided by the mrt layer into discrete messages that can be handled by
other functions. Uses the length at the beginning of each message to
determine how far to read.

2. send_neighbor_msg2(): sends a given message to a neighbor, attaching
message length as described in the format above.

3. neighbor_node_handler(): a thread for each neighbor node, handling all
incoming communication including forwarding data, and broadcasting, etc.
Essentially, receives a message from send_neighbor_msg2(), and passes it
to the appropriate handler for that message type

## Sending Files

### Functions

1. request()
Takes a filename and builds and sends a request message to the neighboring
node that broadcasted the availability of said file.

2. send_file()
Takes a filename and a neighbor. It opens the file, reading n number of bytes
until there is no more to read. It packages it into a json message, and sends
it to a target neighbor.

3. receive_request()
Handles a request message type. If the file is in the local directory, it responds
to the request by calling send_file(). Otherwise, it adds the request to the forward
dictionary and requests the file itself.

4. handle_data_message()
called if received message type is of "data" or "dataf". It checks the forwarding
dictionary to see if the package needs to be forwarded, otherwise it is to be downloaded
itself. The function guarantees that forward requests of the same package but
to different addresses will be fulfilled. 

### Data Flow

When a node wants a file. It calls the request function, which sends a request message.
If a node receives a request message, it either sends the file from its local
directory, or sends its own request message after appending the forward order
to its forward directory. When a node receives data, if the data is to be forwarded,
the message is immediately passed on without being modified. Otherwise, a file with
the message's filename is opened and the message's data is written into the file.

## Additional Features: (4) Efficient Routes and (5) Users Can Chat

## Broadcast and Efficient Routes

Every node as a table that maps every available file to the shortest available path. This table is called the `file_directory`.
As a node, every time we receieve a file, we then `broadcast` out that we have that file, 
which gives other nodes a chance to update the `file_directory` table. Through this, every node is aware of the closest node that has a file,
and because of this, any file transfer takes the fewest possible number of hops.

If we didn't have this feature, nodes may transfer files inefficiently because they don't have up-to-date information about which node is the
closest node that also contains a given file. In this scenario, a file would be transfered _through_ a node which actually could've been the originator of
that file, because that node also has the full download of the file.

The file_directory stores for each filename, the number of nodes that a
broadcast message traversed to reach this node, from the originator, and
the port and ip of the neighbor node that provided us with this file. This
way, when a broadcast comes in, we check the filename, and then check if
the number of hops on the broadcast message is less than the current
minimum stored in the file directory. If it is, then we update our own
file directory as necessary, and pass the broadcast message to all our
neighbors, incrementing the number of hops by one. If it isn't, then this
message and associated path won't be an improvement for our neighbors.
This ensures that files in a cycle of nodes won't be passed around forever.
Further, this assures us that all paths are efficient (as measured by
the number of hops). When a file is announced with the ADD
(broadcast) call, the number of hops is 0, and this information
is passed to all neighbors. When a node joins an already established network,
the 2 nodes that are connecting each share their whole file directories using
the broadcast functionality.

Because each user stores the neighbor address providing the shortest path
for a given file, a file request and download should follow these paths,
so downloads will consequently traverse shortest paths and are therefore
efficient when measured by number of hops.



## Chat

We also implemented chat functionality. 
To designate a message as a "chat" message, we append "chat" to the header.
The chat messgaes travel over the same decentralized network as the files do. 
This creates a few complications because it means that chat messages can reach a node multiple times.
However, we only want each user to see the chat message one time. 
As a solution, we keep a dictionary of _chat count_ which recognize the highest chat # that we have receieve from each node.
Each message has a higher chat number than the last, almost like an auto-incrementinhg id.
Finally, we only _print_ a chat message to a user's terminal if it reaches their message handler and:

1. This is the first time the node is seeing this neighbor chatting
2. the chat from this node has a higher counter value than any previous chats

### Copyright
Copyright (c) 2020 Zonnebloem. [See license](./LICENSE).
