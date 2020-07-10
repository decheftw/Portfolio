# Design Spec for Maze Solver

## Design Spec for `AMStartup.c`

### Overview

`AMStartup.c` is the central program in our Maze solving solution. to solve mazes, we call `AMStartup.c`, which communicates with the server. It sends an initialization message (`AM_INIT`) to the server. 
* Upon receiving the `AM_INIT` messege, the server then generates a maze and returns a messege specify the port for the maze that it generates, the maze's height, the maze's width, as well as confirmation that the server recieved the `AM_INIT` messege. 
* `AMStartup.c` Then goes about setting up a thread of `newavatar.c` for each Avatar. It passes the maze's port number, the avatar's id number, the total number of avatars, and the difficulty of the maze into each instance of `newavatar.c`. 
* From here, it is up to the avatars to find each other as the logic for locating each other in the maze is contained within `newavatar.c`.
* `AMStartup.c` is also responsible for creating a log file where it will print the name of the user of the module, the maze port, and the date and time. Each avatar thread is also responsible for printing the moves the attempt to make in this log file

### User interface
`AMStartup.c`'s only interface with the user is on the command-line; it must always have three arguments.

```
./AMStartup -n numberOfAvatars -d Difficulty -h hostName 
```
`numberOfAvatars` and `Difficulty`, are integers sent through the `AM_INIT` message
* `numberOfAvatars` refers to the number of threads of `newavatar.c` that `AMStartup.c` should create
    * The value should not exceed 10.
* `Difficulty` refers to the difficulty of the maze that `AMStartup.c`is going to ask the server to create
    * The value of `Difficulty` should ot exceed 9.
* `hostName` refers to the name of the host where the maze server is running. For this project, the host should always be `flume.cs.dartmouth.edu`. We are only including the option of a hostname for the sake of flexibilty.

For example:

```
./AMStartup 3 4 flume.cs.dartmouth.edu
```

### Inputs and outputs

#### Inputs: 

The only inputs are command-line parameters; see the User Interface above.

#### Outputs: 

`AMStartup.c` is responsible for printing the Maze GUI(Graphical User Interface) to the terminal.  
`AMStartup.c` is also responsible for creating the log file titled `Amazing_$USER_N_D.log`
* where `$USER` refers to the current user calling `AMStartup.c`,
* `N` refers to the number of avatars in this run of `AMStartup.c`
* `D` refers to the difficulty of the maze in this run of `AMStartup.c`  
`AMStartup.c` prints the first line of the log file, which describes the `$USER` of `AMStartup.c`, the MazePort, and the date and time.  
The threads of `Avatar.c` are responsible for printing the rest. 


### Functional decomposition into modules

1. main, which parses arguments and attempts to connect to server
2. InitializeThreads, which creates the threads of `newavatar.c`. And sets the variables given from the server, calling on functions in newavatar.c
3. MazeGUI functions: AMStartup calls only on
    * newMaze()
    * deleteMaze()


### Pseudo code

AMStartup.c:

    main:
        Parse all the arguments
        Validate inputs
        Set up connection with the server
        Sends AM_INIT message to server
        Reads AM_INIT_OK message from server
        Initializes variables (MazeHeight, MazeWidth, MazePort, Maze, spacesLeft)
        Initializes and writes first line into log file
        Initializes threads
    
    initialize_threads:
        For the number of avatars, start a thread for each one with a unique avatar ID name
        Avatars all send numbers to server, get ready back from server
        Each avatar given turn number and location of each avatar Locations to determine algorithm, turn number to determine when to move



### Data Flow: (data flow through the module)

1. main: recieves parameters from argv and constructs an `AM_INIT` messege, which it sends to the server. It wait for a confirmation from the server before calling `initialize_threads()`

2. createAvatars: called by `main`. creates the avatar threads

3. mazeGUI: prints maze GUI onto terminal using the 2-D array representing the maze. (This may be interactive, i.e. printing while the avatar threads are moving, or it may be from reading the log file)



### Major data structures

1. Maze
    * Fill the Array with actuall ASCii elements (i.e. an array of  strings). This means we record the location of walls as well as print the GUI using   the same array. This implementation has the downside of being complicated to determine if a wall exists at a location.
    * 2 seperate 2-D arrays. One that stores the locations of walls and has a size equivalent to the width of Maze * 2 by the height of Maze * 2. 

### Exit Statuses
exit(1) = incorrect command line inputs  
exit(2) - number of avatars exceeds 10 (maximum) or is below the minimum of 2  
exit(3) - difficulty exceeds 9 (maximum)  
exit(4) - error opening socket  
exit(5) - hostname not functioning  
exit(6) - error connecting socket to the server  
exit(7) - error writing to stream socket  
exit(8) - error reading on the stream socket  
exit(9) - error message received  


## Design Spec for `Avatars.c`

### Overview

newavatar.c holds the code that is the logic for how the avatars proceed through the maze in order to find each other. We used a combination of three heuristics:
1. `deadEndCheck()`: Used when the avatar is surrounded by 3 walls - moves the avatar out of the dead end and adds walls to close off the dead end
2. `rightHandRule()`: Used when the other two heuristics are not viable - always move the avatar to the right.
Throughout its code, newavatar.c calls on maze functions explained below in `toolkit.c`
  

### User interface
`Avatars.c` does not have a user interface; its run_avatar function is called by `AMStartup.c` using threads.

### Inputs and outputs

#### Inputs: 

The only inputs are command-line parameters; see the User Interface above.


#### Outputs: 

Each thread of newvatar.c` is responsible for printing every move it makes to the log file titled `Amazing_$USER_N_D.log`
* where `$USER` refers to the current user calling `AMStartup.c`, which in turn called newavatar.c`
* `N` refers to the number of avatars in this run of `AMStartup.c`
* `D` refers to the difficulty of the maze in this run of `AMStartup.c`.
Each thread also calls `printMaze()` as it runs - printing updated versions of the maze



### Functional decomposition into modules

#### functions called in AMStartup.c to set values on avatar struct
1. `void avStatus_t* avStatus_new();`
2. `void avStatus_setAvatarId(avStatus_t* stat, int id);`
3. `void avStatus_setnAvatars(avStatus_t* stat, int nAvatars);`
4. `void avStatus_setdifficulty(avStatus_t* stat, int difficulty);`
5. `void avStatus_sethostname(avStatus_t* stat, char* hostname);`
6. `void avStatus_setmazePort(avStatus_t* stat, int mazePort);`
7. `void avStatus_setfilename(avStatus_t* stat, char* filename);`
8. `void avStatus_setmaze(avStatus_t* stat, maze_t* maze);`
9. `void avStatus_setspacesLeft(avStatus_t* stat, int* spacesLeft);`
10. `void avStatus_setlock(avStatus_t* stat, pthread_mutex_t* lock);`
11. `void avStatus_delete(avStatus_t* stat);`

#### functions called in newavatar.c for algorithm logic

1. `void* run_avatar(void* av1);`
2. `void updatePosition(avStatus_t* av, int comm_sock)`
3. `void move(int dir, avStatus_t* av, int comm_sock)`
4. `void backTrackme(avStatus_t* av, int comm_sock)`
5. `int oppositeDirection(int dir)`  
6. `int getRight(int dir)` 
7. `int getLeft(int dir)`
8. `void rightHandRule(avStatus_t* av, int comm_sock)` 
9. `void logging(avStatus_t *av, const char *str)`



### Data Flow: (data flow through the module)

`run_avatar()` runs while loop until maze is solved, calling on helper functions and passing avatar struct information to the functions, including:

* `updatePosition()` - updates current position of avatar and normalizes positions of all avatars  
* `rightHandRule()` - optimizes moving avatar right, then forward, then left
* `move()` - takes avatar, direction, and comm_sock, validates the parameters given, and sends a message to the server
* `getRight()` - given a int direction, returns a new direction so the avatar has now turned right  
* `getLeft()` - given a int direction, returns a new direction so the avatar has now turned left 
* `logging()` - updates log file when an avatar succesfully moves 
* `backTrackme()` - runs deadEndCheck heuristic  
* `oppositeDirection()` - given an int direction, returns an int direction that is the opposite



### Major data structures

1. Avatar Struct: avStatus  
variables given by server:

        int AvatarId;  
        int nAvatars;  
        int difficulty;  
        char *hostname;  
        int mazePort;  
        char *filename;  

variables updated within newavatar.c:  

        hashtable_t *ht;           // Hashtable of hashtables (breadcrumb trails of each avatar)
        int direction;             // Direction avatar is facing
        maze_t *maze;              // Maze gui and wall location storage
        XYPos *curr;               // Current point
        XYPos *prev;               // Previous point
        pthread_mutex_t* lock;     // Mutex lock variable
        AM_Message response;       // server response message
        int* spacesLeft;           // Stores number of spaces left 

2. Also uses MazeGUI described above


### Pseudo code for logic/algorithmic 

`run_avatar()` initializes connection to socket  
    initializes local avStatus struct variables  
    sends AM_AVATAR_READY message to server  
    receives AM_AVATAR_TURN message from server  
    updates current position  
    normalizes all positions sent from server  
    sets avatar onto maze and prints maze  

`run_avatar()` loops until maze has been solved  
check server message type  
if it is this thread's turn  
if there are less than 4 spaces left not walled off, use consolidate avatars heuristic then update position  
if you are surrounded by 3 walls, use `deadEndCheck()` heuristic
otherwise, use `rightHandRule()` heuristic  


Once maze is solved, free avStatus struct and prev point


consolidate avatars heuristic:  

If global variable `spacesLeft` is less than 4,  
then we set a final point at the curr point of avatar 0  
and all avatars use a random move and stop moving once they reach the final point  


`deadEndCheck()` heuristic:  

check if the avatar is in a dead end and sets direction based off that (calling `deadEndCheck()` located in `toolkit.c`)  
do not add a wall if there is another avatar in the dead end with you
otherwise, add a wall (using `oppositeDirection()`) - this area is now blocked off
decrement the number of open spaces left (`av->spacesLeft`)  
update the display  


`rightHandRule()` heuristic:  Attempt to move right, then forward, then left

if there is no wall to the right of the avatar, move right (calling `getRight()` and `move()`)  
if the avatar successfully moved right, update `av->direction` appropriately so that it turns right (calling `getRight()`)  
otherwise, add a wall to the right if it couldn't successfully move right (calling `getRight()`)  


if there is a wall to the right of the avatar, move forward (checking if there is a wall in front of it first) (calling `move()`)  
if it did not successfully move, add a wall in front of it and turn left (calling `getLeft()`)  
if there is a wall in front of the avatar, turn direction left (calling `getLeft()`)  


if the avatar succefully moves, it calls on `logging()` to update the MazeGUI



## Design Spec for `toolkit.c`

### Overview

Toolkit.c holds helper functions used in both AMStartup.c and newavatar.c as well as the struct definition for the maze struct

### Functions

`int checktype(uint32_t type)`  
used to validate AM_INIT_OK, AM_AVATAR_TURN, and AM_MAZE_SOLVED messages  


#### Maze Functions

`maze_t* newMaze(int width, int height)`  
creates a maze  


`void addWall(maze_t* maze, XYPos* from, int dir)`  
adds a wall from "from" in given dir  


`void addWallBacktrack(maze_t* maze, XYPos* from, int dir)`  
a variation of add wall, that only adds a wall to the "matrix" maze, not that GUI (for aesthetics)   


`void setAvatar(maze_t* maze, char id, XYPos* loc)`  
sets the representation of the avatar in the correct location  


`void printMaze(maze_t* maze)`  
prints maze  


`bool wallCheck(maze_t* maze, XYPos* from, int dir)`  
returns true if there is a wall from "from" in the given "dir", false if not  


`int deadEndCheck(maze_t* maze, XYPos* loc)`  
if the given "loc" is surrounded by 3 walls, returns the free direction  
otherwise returns -1  


`bool deleteMaze(maze_t* maze)`  
frees memory space of maze  


## Testing plan for overall module

See `Testing.md` for an explanation of how we tested.


## Kinds of messages passed between server and client  

Message Type  --->  Summary  


AM_INIT                     -->  asks server to setup a new maze  
AM_AVATAR_READY            	-->  tells server that a new Avatar is ready to move  
AM_AVATAR_MOVE              -->  tells server where an Avatar wishes to move  
AM_INIT_OK                  -->  response: initialization succeeded  
AM_INIT_FAILED              -->  response: initialization failed  
AM_NO_SUCH_AVATAR           -->  response: referenced an unknown or invalid Avatar  
AM_AVATAR_TURN              -->  response: updated Avatar (x,y) position and proceed to next turn  
AM_MAZE_SOLVED              -->  response: the maze was solved  
AM_UNKNOWN_MSG_TYPE         -->  response: unrecognized message type  
AM_UNEXPECTED_MSG_TYPE      -->  response: message type out of order  
AM_AVATAR_OUT_OF_TURN       -->  response: Avatar tried to move out of turn  
AM_TOO_MANY_MOVES           -->  response: exceeded the max number of moves  
AM_SERVER_TIMEOUT           -->  response: exceeded time between messages  
AM_SERVER_DISK_QUOTE        -->  response: server has exceeded disk quota  
AM_SERVER_OUT_OF_MEM        -->  response: server failed to allocate memory  
