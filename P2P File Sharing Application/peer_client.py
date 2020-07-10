import sys
import os
from collections import namedtuple
from node import Node, NeighborNode
import stun

# TODO:
#   1. QUIT doesn't work (probably need to tear down the server thread)?
#   2. Input sanitization (write error messages)

Command = namedtuple("Command", ["args", "description", "handler"])
class PeerClient:
    def __init__(self, port, directory, private):
        self.node = Node.start_node(port, directory, private)
        self.commands = {
            "CONNECT": Command(["ip", "port"], 
                               "Connect to the given ip/port combination", 
                               self.connect_handler), 
            "ADD":     Command(["file"],
                               "Add the given file to the network. Must be in the given directory.", 
                               self.add_handler),
            "GET":     Command(["file"], 
                               "Get the named file from the network.",
                               self.get_handler),
            "LIST":    Command([], 
                               "List the names of available files in the network.",
                               self.list_handler),
            "IP":    Command([], 
                               "Get the external IP Address of your node.",
                               self.ip_handler),
            "CHAT":    Command(["text"], # list of string
                               "Send the text to other users on the network.",
                               self.chat_handler),
            "HELP":    Command([],
                               "Display this help message again.",
                               self.help_handler),
            "QUIT":    Command([],
                               "Exit the program.", None
                               ),
        }
    
    def get_username(self):
        while True:
            name = input("Enter username: ")

            if not name.isalnum():
                print("All characters in name must be alphanumeric.")
                continue

            return name

    # Event loop for the peer client
    def start(self):
        self.node.username = self.get_username()
        self.help_handler([])

        while True:
            user_input = input("> ")
            user_input_sp = user_input.split(" ")
            command, args = user_input_sp[0], user_input_sp[1:]
    
            if command not in self.commands:
                print("Command '{}' not recognized".format(command))
                continue
    
            if command == "QUIT":
                self.quit_handler([])
                return
    
            handler = self.commands[command].handler
            handler(args)
    
    def connect_handler(self, args):
        if len(args) != 2:
            print("Invalid arguments. Correct usage:")
            print("CONNECT [ip] [port]")
            return

        ip = args[0]
        port = int(args[1])
    
        params = (ip, port)
        if ip == self.node.external_ip:
            user_input = input("Please enter target's local IP and port: ")
            user_input_sp = user_input.split(" ")

            local_ip = user_input_sp[0]
            local_port = int(user_input_sp[1])

            params = (ip, port, local_ip, local_port)

        try:
            self.node.connect(*params) # unrolls the params
            print("Connected Successfully!")
        except:
            print("Invalid IP Address provided.")
    


    def add_handler(self, args):
        if len(args) != 1:
            print("Invalid arguments. Correct usage:")
            print("ADD [file]")
            return
        
        file = args[0]
    
        if file not in os.listdir(self.node.storage):
            print("Error: file '{}' not found in directory '{}'.".format(file, self.node.storage))
            return

        if file in self.node.file_directory:
            print("Error: file '{}' already exists in the network. Cannot add file with the same name.".format(file))
            return

        self.node.add_file(file)
    
    def get_handler(self, args):
        if len(args) != 1:
            print("Invalid arguments. Correct usage:")
            print("CHAT [message]")
            return
    
        file = args[0]
        
        if file in os.listdir(self.node.storage):
            print("Error: file '{}' exists locally. Cannot get over the network.".format(file))
            return
    
        if file not in self.node.file_directory:
            print("Error: file '{}' doesnt not exist in the network".format(file))
            return

        self.node.request(file)
    
    def list_handler(self, _):
        print("Files available:")
        for file in self.node.file_directory:
            print("\t" + "* " + file)
    
    def ip_handler(self, _):
        print("Fetching IP and Port...")
        
        nat_type, external_ip, external_port = stun.get_ip_info()
        print("Connection Info:")
        print("My NAT Type: " + nat_type)
        print("My IP: " + external_ip)
        print("My External Port: " + str(external_port))
    
    def chat_handler(self, args):
        if len(args) == 0:
            print("Invalid arguments. Correct usage:")
            print("ADD [file]")
            return

        # Recreate a space delimited string
        text = " ".join(args)
    
        self.node.create_chat_msg(text)
    
    def help_handler(self, _):
        help_msg = []
        for name, command in self.commands.items():
            args = " ".join("[" + arg + "]" for arg in command.args)
    
            help_msg.append(name + " " + args + "\n\t" + command.description)
            
        print("\n".join(help_msg))

    def quit_handler(self, _):
        self.node.node_close()

    
# Argument parsing
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 peer_client.py [port] [directory] [--public]")
        sys.exit(1)
    port = int(sys.argv[1]) 
    directory = sys.argv[2]

    # private=True by default
    # if the user sets the --public option then set private=False
    private = True
    if len(sys.argv) == 4:
        private = not (sys.argv[3] == "--public")

    peer = PeerClient(port, directory, private)
    peer.start()
