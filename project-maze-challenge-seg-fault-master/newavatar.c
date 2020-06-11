/* GROUP: Seg Fault
 * COSC 50, Winter 2020
 * Avatar code
 *
 * See avatar.h for more info on the functions defined here.
 *  
 */

#include "avatar.h"

/******************************* Avatar status struct ************************************/
/*
 * This is the struct responsible for storing the information that 
 * each individual struct needs to solve the maze.
 */
typedef struct avStatus {
    /* Server Variables */
    int AvatarId;
    int nAvatars;
    int difficulty;
    char *hostname;
    int mazePort;
    char *filename;

    /* variables we created */
    int direction;             // Direction avatar is facing
    maze_t *maze;              // Maze gui and wall location storage
    XYPos *curr;               // Current point
    XYPos *prev;               // Previous point
    XYPos* finalLocation;       // Final Location
    pthread_mutex_t* lock;     // Mutex lock variable
    AM_Message response;       // server response message
    int* spacesLeft;           // Stores number of spaces left 
} avStatus_t;


/********************* Avatar status struct's associated functions ***************************/

/**************************avStatus_new***************************/
/*****************************************************************/
avStatus_t* avStatus_new()
{
    // Malloc the struct
    avStatus_t* stat = malloc(sizeof(avStatus_t));

    if (stat == NULL) {                                         // Malloc Check
        fprintf(stderr, "error allocating memory for avStatus");
        return NULL; 
    }
    else {
        return stat;
    }
}

/*********************avStatus_setAvatarId************************/
/*****************************************************************/
void avStatus_setAvatarId(avStatus_t* stat, int id)
{
    stat->AvatarId = id;           // Assign the struct's avatar id
}

/*********************avStatus_setnAvatars************************/
/*****************************************************************/
void avStatus_setnAvatars(avStatus_t* stat, int nAvatars)
{
    stat->nAvatars = nAvatars;     // Assign the struct's  total number of avatars
}

/********************avStatus_setdifficulty***********************/
/*****************************************************************/
void avStatus_setdifficulty(avStatus_t* stat, int difficulty)
{
    stat->difficulty = difficulty;  // stores the difficulty of the current maze 
                                    // that the avatar is attempting to solve
}

/**********************avStatus_sethostname***********************/
/*****************************************************************/
void avStatus_sethostname(avStatus_t* stat, char* hostname)
{
    stat->hostname = hostname;      // Stores the host name
}

/**********************avStatus_setmazePort***********************/
/*****************************************************************/
void avStatus_setmazePort(avStatus_t* stat, int mazePort)
{
    stat->mazePort = mazePort;      // Stores the maze's port for later use 
}

/**********************avStatus_setfilename***********************/
/*****************************************************************/
void avStatus_setfilename(avStatus_t* stat, char* filename)
{
    stat->filename = filename;       // Stores filename
}

/***********************avStatus_setmaze**************************/
/*****************************************************************/
void avStatus_setmaze(avStatus_t* stat, maze_t* maze)
{
    stat->maze = maze;               // Stores the maze struct
}

/*********************avStatus_setspacesLeft**********************/
/*****************************************************************/
void avStatus_setspacesLeft(avStatus_t* stat, int* spacesLeft)
{
    stat->spacesLeft = spacesLeft; // Variable for storing the 
                                       // number of avatars that are 
                                       // following any particular avatar
}

/************************avStatus_setlock*************************/
/*****************************************************************/
void avStatus_setlock(avStatus_t* stat, pthread_mutex_t* lock)
{
    stat->lock = lock;                 // Stores the mutex lock variable
}

/*********************avStatus_setfinalPoint**********************/
/*****************************************************************/
void avStatus_setfinalPoint(avStatus_t* stat, XYPos* loc){
    stat->finalLocation = loc;                  // Stores the final location variable
}

/************************avStatus_delete**************************/
/*****************************************************************/
void avStatus_delete(avStatus_t* stat)
{
    // Null Check
    if (stat != NULL) {
        free(stat);                     // Free the struct if it is not null
    }
    else {
        fprintf(stderr, "error: deleting a null avStatus");
    }
}

/************** Avatar.c's main thread function and its associated functions ****************/

/*****************************run_avatar**************************/
/*****************************************************************/
void* run_avatar(void* av1) 
{
    
    avStatus_t* av = av1;                 // Stores the information passed in from AMStartup into a local struct variable
    
    /* Thread Setup Operations *************************************************/

    // Initialize the fields of the server address using mazeport passed in
    int comm_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (comm_sock < 0) {
        perror("opening socket");
        return NULL;
    }
    struct sockaddr_in server;             // address of the server
    server.sin_family = AF_INET;         
    server.sin_port = htons(av->mazePort); // Stores maze port


    // Look up the hostname specified on command line
    struct hostent *hostp = gethostbyname(av->hostname);       // server hostname
    if (hostp == NULL) {                                       // Null check
        fprintf(stderr, "unknown host '%s'\n", av->hostname);
        return NULL;
    }

    memcpy(&server.sin_addr, hostp->h_addr_list[0], hostp->h_length); // allocate memory for smth i think (i didnt do this part)

    // Connect the socket to that server   
    if (connect(comm_sock, (struct sockaddr *) &server, sizeof(server)) < 0) {
        perror("connecting stream socket");
        return NULL;
    }

    // Initialize avStatus struct variables that are local to this thread 
    av->direction = M_NORTH;     // Arbitrary initialize direction
    av->prev = NULL;
    av->curr = NULL;
    
    // Send the AM_AVATAR_READY messege to the server now that we have successfully connected
    AM_Message message;
    memset(&message, 0, sizeof(message));         // clear up the buffer
    int id = av->AvatarId;
    message.avatar_ready.AvatarId = htonl(id);    // Tell the server which avatar that this messege is coming from
    message.type = htonl(AM_AVATAR_READY);        // Tell the server what kind of messege
    
    if (write(comm_sock, &message, sizeof(message)) < 0) { // send AM_AVATAR_READY message to server
        perror("writing on stream socket");
        return NULL;
    }

    // Read the server's response to our initial AM_AVATAR_READY messege
    if (read(comm_sock, &av->response, sizeof(av->response)) < 0) { // read AM_AVATAR_TURN message in from server
        perror("reading on stream socket");
        return NULL;
    }

    // Extract current location from the server's response
    av->curr = &av->response.avatar_turn.Pos[av->AvatarId];

    // Normalize positions for all avatars
    for (int i = 0; i < av->nAvatars; i++) { 
        XYPos *temp = &av->response.avatar_turn.Pos[i];
        temp->x = ntohl(temp->x);
        temp->y = ntohl(temp->y);
    }
    
    // Place avatar on maze in preperation for its journey 
    char c = av->AvatarId + '0';             // Convert AvatarId from being an int to being a char
    setAvatar(av->maze, c, av->curr);        // Set the avatar into the maze's Gui
    pthread_mutex_lock(av->lock);            // Mutex lock so that only one maze is printed at a time
    printMaze(av->maze);                     // Print the maze
    pthread_mutex_unlock(av->lock);          // Mutex unlock so that only one maze is printed at a time

    /* Run the maze Heuristic until the maze has been solved*******************************/
    while(1){ 
        // Check if the server has sent an error message or the maze is solved
        if (checktype(av->response.type) == -1) {                              // error message received
            break;    
        }
        if (checktype(av->response.type) == 2) {                               // if it's an AM_MAZE_SOLVED message
            printf("Maze solved!\n");
            char str[100];
            sprintf(str, "maze solved: %d", ntohl(av->response.maze_solved.Hash));
            logging(av, str);
            break;
        }

        // Check if it is the current avatar's turn
        if (checktype(av->response.type) == 1 && ntohl(av->response.avatar_turn.TurnId) == av->AvatarId) {
            /* First Heuristic: Are we about finish the game? If yes, we consolidate the avatars ************************/
            if((*av->spacesLeft) < 4){                  // If we are in the end stages of the game
              
                // Make sure your info is up to date
                if(av->AvatarId == 0){
                    av->finalLocation->x = av->curr->x;
                    av->finalLocation->y = av->curr->y;
                }

                if((av->curr->x == av->finalLocation->x) && (av->curr->y == av->finalLocation->y)){
                    move(8, av, comm_sock);               // Stop moving the point
                    updatePosition(av, comm_sock);
                    char c = av->AvatarId + '0';                        // Convert AvatarId to a char
                    pthread_mutex_lock(av->lock);                       // Mutex lock so that only one thread can alter graphics at a time
                    setAvatar(av->maze, ' ', av->prev);                 // Remove the avatar graphic from the previous location
                    setAvatar(av->maze, c, av->curr);                   // Place the avatar graphic into a new location
                    printMaze(av->maze);                                // Print the maze
                    pthread_mutex_unlock(av->lock);                     // Unlock the mutex so that other threads may function
            
                    continue;
                }
                
                printf("tryna get to where im going %d\n", av->AvatarId);
                // If you are not yet at the final wall RHR your way there
                int m = rand()%4;                                   // Give the avatar a random direction
                if(wallCheck(av->maze, av->curr, m)){
                    move(8, av, comm_sock);
                    updatePosition(av, comm_sock);
                    continue;
                }
                move(m, av, comm_sock);
                updatePosition(av, comm_sock);
                char c = av->AvatarId + '0';                        // Convert AvatarId to a char
                pthread_mutex_lock(av->lock);                       // Mutex lock so that only one thread can alter graphics at a time
                setAvatar(av->maze, ' ', av->prev);                 // Remove the avatar graphic from the previous location
                setAvatar(av->maze, c, av->curr);                   // Place the avatar graphic into a new location
                printMaze(av->maze);                                // Print the maze
                pthread_mutex_unlock(av->lock);                     // Unlock the mutex so that other threads may function
                
                
                
            }

            /* Second Heuristic: Is the current avatar Surrounded by three walls? ***************************************/
            else if(deadEndCheck(av->maze, av->curr) > -1){
                // Avatar is in a dead end. Move it backwards and cllose of the dead end
                backTrackme(av, comm_sock);
            }

            /* Third Heuristic: If it is not appropriate to use the first or second heuristic, use the Right Hand Rule ***/
            else{
                rightHandRule(av, comm_sock);
            }

        }

        // If it is not this avatar's turn, make sure it is still updating its info.
        else{
            updatePosition(av, comm_sock);
        }
    }

    // Maze is Solved, free all the stuffs
    if(av->prev != NULL){
        free(av->prev);
    }
    if(av != NULL){
        avStatus_delete(av);
    }
    

    return NULL;
}

/************************updatePosition***************************/
/*****************************************************************/
void updatePosition(avStatus_t* av, int comm_sock){
    // Read the server message
    if (read(comm_sock, &av->response, sizeof(av->response)) < 0) { // read AM_INIT_OK message in from server
        perror("reading on stream socket");
    }

    // Check the message type
    if (checktype(av->response.type) == 1) {

        // Update current positions
        av->curr = &av->response.avatar_turn.Pos[av->AvatarId];

        // Convert server giberish into normal giberish
        for (int i = 0; i < av->nAvatars; i++) { 
            XYPos *temp = &av->response.avatar_turn.Pos[i];
            temp->x = ntohl(temp->x);
            temp->y = ntohl(temp->y);
        }
    }
}

/******************************move*******************************/
/*****************************************************************/
void move(int dir, avStatus_t* av, int comm_sock) {

    // If this thread does not have a prev yet
    if (av->prev == NULL) {
        av->prev = malloc(sizeof(XYPos));
    }

    // Invalid direction check
    if (dir < 0) {
        fprintf(stderr, "dir is negative: %d\n", dir);
        return;
    }

    char str[100];
    sprintf(str, "Avatar %d is currently at %d, %d and is moving towards %d", av->AvatarId, av->curr->x, av->curr->y, dir);
    logging(av, str);
    // Position updates
    int x = av->curr->x;
    int y = av->curr->y;
    av->prev->x = x;
    av->prev->y = y;

    // Create the message that we are gonna send to the server
    AM_Message message;
    memset(&message, 0, sizeof(message));                        // clear up the buffer
    // setting the message variables
    message.avatar_move.Direction = htonl(dir);
    message.avatar_move.AvatarId = htonl(av->AvatarId);
    message.type = htonl(AM_AVATAR_MOVE);

    // Actually write the message to the server
    if (write(comm_sock, &message, sizeof(message)) < 0) {       // send AM_AVATAR_MOVE message to server
        perror("writing on stream socket");
        return;
    }

    // Sleep after moving 
    // sleep(1);
}

/**************************backTrackme****************************/
/*****************************************************************/
void backTrackme(avStatus_t* av, int comm_sock){
    av->direction = deadEndCheck(av->maze, av->curr);
    
    // Do not add wall if other avatars are backtracking with you
    int numberOfAdjacent = 0;

    for (int i = 0; i < av->nAvatars; i++) { 
        // Obviously dont check urself
        if(i != av->AvatarId){
            XYPos *temp = &av->response.avatar_turn.Pos[i];
            
            // There is an avatar where we just left. DO NOT ADD WALL
            if((temp->x == av->curr->x) && (temp->y == av->curr->y)){
                numberOfAdjacent++;
            }
        }
    }

    move(av->direction, av, comm_sock);
    updatePosition(av, comm_sock);
    
    if((numberOfAdjacent == 0) && ((av->prev->x != av->curr->x) || (av->prev->y != av->curr->y))){
        addWallBacktrack(av->maze, av->curr, oppositeDirection(av->direction)); 
        char str[100];
        sprintf(str, "Avatar %d has discovered a wall", av->AvatarId);
        logging(av, str);
        
        // If you have succesfully blocked of an area, update the number of spaces left
        (*av->spacesLeft)--;
    }
    
    // Display
    char c = av->AvatarId + '0';
    pthread_mutex_lock(av->lock);
    setAvatar(av->maze, ' ', av->prev);
    setAvatar(av->maze, c, av->curr);
    printMaze(av->maze);
    pthread_mutex_unlock(av->lock);
}

/*********************oppositeDirection***************************/
/*****************************************************************/
int oppositeDirection(int dir){
    if(dir == 0){
        return 3;
    }
    else if(dir == 3){
        return 0;
    }
    else if(dir == 1){
        return 2;
    }
    else if(dir == 2){
        return 1;
    }
    
    return -1;         // Direction that was entered is invalid
}

/****************************getRight*****************************/
/*****************************************************************/
int getRight(int dir){
    int result;
    if (dir > 3 || dir < 0){
        printf("wrong x: %d\n", dir);
        return -1;
    }
    if (dir == 3) {
        result = 2;
    }
    else if (dir == 2) {
        result = 0;
    }
    else if (dir == 0) {
        result = 1;
    }
    else {
        result = 3;
    }
    return result;
}

/****************************getLeft******************************/
/*****************************************************************/
int getLeft(int dir){
    int result;
    if (dir > 3 || dir < 0){
        printf("wrong x: %d\n", dir);
        return -1;
    }
    if (dir == 3) {
        result = 1;
    }
    else if (dir == 2) {
        result = 3;
    }
    else if (dir == 0) {
        result = 2;
    }
    else {
        result = 0;
    }
    return result;
}

/**************************rightHandRule**************************/
/*****************************************************************/
void rightHandRule(avStatus_t* av, int comm_sock){
    // If there's no wall on the right of the avatar, move the avatar right
    if (!wallCheck(av->maze, av->curr, getRight(av->direction))){

        move(getRight(av->direction), av, comm_sock);       // Send the move messege to the server
        updatePosition(av, comm_sock);                      // Read the Server's response and update position
        char c = av->AvatarId + '0';                        // Convert AvatarId to a char
        pthread_mutex_lock(av->lock);                       // Mutex lock so that only one thread can alter graphics at a time
        setAvatar(av->maze, ' ', av->prev);                 // Remove the avatar graphic from the previous location
        setAvatar(av->maze, c, av->curr);                   // Place the avatar graphic into a new location
        printMaze(av->maze);                                // Print the maze
        pthread_mutex_unlock(av->lock);                     // Unlock the mutex so that other threads may function
        
        // If the avatar has successfully moved right, update its direction appropriatley
        if ((av->curr->x != av->prev->x) || (av->curr->y != av->prev->y)) {
            av->direction = getRight(av->direction);
        }
        // if the avatar has unsuccessfully move right, add a wall to the right of it and proceed
        else {
            addWall(av->maze, av->curr, getRight(av->direction));  // Add a wall to the right of the avatar
             
            // Append to log file
            char str[100];
            sprintf(str, "Avatar %d has discovered a wall", av->AvatarId);
            logging(av, str);

            av->direction = av->direction;                         // Direction does not change because
        }
    }
    // If there is a wall on the right of the avatar, move the avatar forward
    else if (wallCheck(av->maze, av->curr, getRight(av->direction))) { 
        // Check to see if we know that there is a wall infront of the avatar. If we do not think there is one:
        if (!wallCheck(av->maze, av->curr, av->direction)) {
            move(av->direction, av, comm_sock);                    // Send the move messege to the server
            updatePosition(av, comm_sock);                         // Read the Server's response and update position
            char c = av->AvatarId + '0';                           // Convert AvatarId to a char
            pthread_mutex_lock(av->lock);                          // Mutex lock so that only one thread can alter graphics at a time
            setAvatar(av->maze, ' ', av->prev);                    // Remove the avatar graphic from the previous location
            setAvatar(av->maze, c, av->curr);                      // Place the avatar graphic into a new location
            printMaze(av->maze);                                   // Print the maze
            pthread_mutex_unlock(av->lock);                        // Unlock the mutex so that other threads may function
            
            // If the avatar has successfully moved forward, update its direction appropriatley
            if ((av->curr->x != av->prev->x) || (av->curr->y != av->prev->y)) {
                // No update to direction
            }
            // If the avatar has unsuccessfully moved forward, add wall in front of it and turn left
            else {
                addWall(av->maze, av->curr, av->direction); 
                char str[100];
                sprintf(str, "Avatar %d has discovered a wall", av->AvatarId);
                logging(av, str);
                av->direction = getLeft(av->direction);        // turn left
            }
        }
        // If we know there's a wall in front of the avatar, turn the avatar left
        else {
            av->direction = getLeft(av->direction); // turn left 
        }
    }
}

void logging(avStatus_t *av, const char *str)
{
    FILE *fp = fopen(av->filename, "a");
    fprintf(fp, "%s\n", str);
    fclose(fp);
}
