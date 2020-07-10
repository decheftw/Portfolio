# project_maze_challenge
# CS50 Winter 2020, Final Project

## Team name: Seg Fault
## Team Members: Alex Feng, Gao Chen, Abubakar Kasule, Katherine Taylor

GitHub usernames: alxfngg, AbubakarKasule, decheftw, kattaylor22

To build, run `make`.

To clean up, run `make clean`.

*Please replace this text with any comments you have about the overall project.*

User interface
`AMStartup.c`'s only interface with the user is on the command-line; it must always have three arguments.

```
./AMStartup -n numberOfAvatars -d Difficulty -h hostName 
```

## AMStartup.c:

    main:
        Parse all the arguments
        Validate inputs
        Set up connection with the server
        The hostname is an argument
        Reads what the server sends back and see if it’s an error or not
        Sends starting message to server
        Server sends maze port and width and height back
        call createAvatars.
        calls mazeGUI.
    
    createAvatars:
        For the number of avatars, start a thread for each one with a unique avatar ID name
        Avatars all send numbers to server, get ready back from server
        Each avatar given turn number and location of each avatar Locations to determine algorithm, turn number to determine when to move

    mazeGUI:
        loop through data structure storing the maze and print it to terminal
        
to build the executable, run make

## Assumptions:
1. the server works as described
2. the mazes are acyclic

