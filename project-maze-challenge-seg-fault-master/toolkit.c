// Seg Fault toolkit for maze solver        
#include "toolkit.h"


/**************** checktype ****************/
int checktype(uint32_t type) {
  uint32_t htype = ntohl(type); // convert type to host byte form
  if (IS_AM_ERROR(htype)) { // error response received
    fprintf(stderr, "received error response: %d\n", htype); // print the error response
    return -1;
  }
  if (htype == AM_INIT_OK) {
    return 0;
  }
  if (htype == AM_AVATAR_TURN) {
    return 1;
  }
  if (htype == AM_MAZE_SOLVED) {
    return 2;
  }
  return 3;
}

// maze struct
typedef struct maze{
    char** GUI;           // 2D array of chars for visualizing the maze
    int** matrix;         // 2D array of ints for storing info about the maze
    int width;
    int height;
}maze_t;

//// Helper functions for the structs

/****************************newMaze******************************/
/*****************************************************************/
maze_t* newMaze(int width, int height){
    maze_t* maze = malloc(sizeof(maze_t));

    // Null checks / Parameter checks
    if ((maze == NULL) || (width < 0 || height < 0)){ 

        return NULL;
    }
    else{
        maze->width = width;
        maze->height = height;

        // Thing to help me visualize lol
        /*
        +---+---+
        |       |
        +   +   +
        |       |
        +---+---+
        */

        // Initialize array of rows
        maze->GUI = malloc(sizeof(char*) * (height * 2 + 1));

        // malloc the lines

        // Initialize blank matrix visualization
        for(int i = 0; i < (height * 2 + 1); i++){
            // for first and last lines
            if (i == 0 || i == (height * 2)){
                char* line = malloc(sizeof(char) * (width * 4 + 2));
                char* temp = "---+";
                line[0] = '+';
                line[1] = '\0';

                for(int j = 0; j < width; j++){
                    strcat(line, temp);
                }
                
                maze->GUI[i] = line;
            }
            else if(i % 2 == 0){
                char* line = malloc(sizeof(char) * (width * 4 + 2));
                char* temp = "   +";
                line[0] = '+';
                line[1] = '\0';

                for(int j = 0; j < width; j++){
                    strcat(line, temp);
                }
                
                maze->GUI[i] = line;
            }
            else{
                char* line = malloc(sizeof(char) * (width * 4 + 2));
                char* temp = "    ";
                line[0] = '|';
                line[1] = '\0';

                for(int j = 0; j < width - 1; j++){
                    strcat(line, temp);
                }

                // One tiny fix to the end of these lines
                temp = "   |";
                strcat(line, temp);
                maze->GUI[i] = line;
            }
              
        }

        maze->matrix = malloc(sizeof(int*) * ((height * 2) - 1));

        for(int i = 0; i < (height * 2) - 1; i++){
            maze->matrix[i] = malloc(sizeof(int) * ((width * 2) - 1));
        }

        //// Initialize all components
        // 0 - empty/unvisited spot
        // -1 - no wall
        // -2 - wall
        /*
        0  -1  0 
        -1 -1  -1
        0  -1  0 
        2 * 2 matrix. Row major
        */
        for(int i = 0; i < (height * 2) - 1; i++){
            for(int j = 0; j < (width * 2) - 1; j++){
                if(i % 2 == 1){
                    maze->matrix[i][j] = -1;
                }
                else{
                    if (j % 2 == 0){
                        maze->matrix[i][j] = 0;
                    }
                    else{
                         maze->matrix[i][j] = -1;
                    }
                }
            }
        }
    }
    return maze;
}

/****************************addWall******************************/
/*****************************************************************/
void addWall(maze_t* maze, XYPos* from, int dir){

    /* Defensive programing ********************************/
    // Null check
    if (maze == NULL || from == NULL){ 
        return;
    }

    // safety checks for from position
    if ((from->x > maze->width || from->y > maze->height) || (from->x < 0 || from->y < 0)) {
        return;
    }
    // saftey checks for direction
    if(dir < 0 || dir > 3){ return;}

    // Edge wall check. because we do not want to add edge walls ofc
    if(dir == 0 && from->x == 0){
        return;
    }
    else if(dir == 1 && from->y == 0){
        return;
    }
    else if(dir == 2 && from->y == maze->height - 1){
        return;
    }
    else if (dir == 3 && from->x == maze->width - 1){
        return;
    }

    /* Do the stuff programing ********************************/
    // Make the 'to' point
    XYPos* to = malloc(sizeof(XYPos));

    // Malloc check
    if(to == NULL){return;}

    if(dir == 0){
        to->x = from->x - 1; to->y = from->y;
    }
    else if(dir == 1){
        to->x = from->x; to->y = from->y - 1;
    }
    else if(dir == 2){
        to->x = from->x; to->y = from->y + 1;
    }
    else{
        to->x = from->x + 1; to->y = from->y;
    }

    //// Add wall to gui
    // Is it a verticle or horizontal wall
    if((int)from->y == (int)to->y){
        if((int)from->x - (int)to->x > 0){
            // It is a verticle wall encountered from its right
            maze->GUI[1 + (2 * (int)from->y)][4 * ((int)to->x + 1)] = '|';
        }
        else{
            // It is a verticle wall encountered from its left
            maze->GUI[1 + (2 * (int)from->y)][4 * ((int)from->x + 1)] = '|';
        }
    }
    else{
        if((int)from->y - (int)to->y > 0){
            // it is a horizontal wall encountered from its bottom
            for(int i = 1; i < 4; i++){
                maze->GUI[2 + (2 * (int)to->y)][i + (4 * (int)to->x)] = '-';
            }
            
        }
        else{
            // it is a horizontal wall encountered from its top
            for(int i = 1; i < 4; i++){
                maze->GUI[2 + (2 * (int)from->y)][i + (4 * (int)to->x)] = '-';
            }
        }
    }

    // Add wall to matrix
    if((int)from->y == (int)to->y){
        if((int)from->x - (int)to->x > 0){
            // It is a verticle wall encountered from its right
            maze->matrix[(2 * (int)from->y)][1 + (2 * (int)to->x)] = -2;
        }
        else{
            // It is a verticle wall encountered from its left
            maze->matrix[(2 * (int)from->y)][1 + (2 * (int)from->x)] = -2;
        }
    }
    else{
        if((int)from->y - (int)to->y > 0){
            // it is a horizontal wall encountered from its bottom
            maze->matrix[1 + (2 * (int)to->y)][(2 * (int)from->x)] = -2;
        }
        else{
            // it is a horizontal wall encountered from its top
            maze->matrix[1 + (2 * (int)from->y)][(2 * (int)from->x)] = -2;
        }
    }

    free(to);
}

/************************addWallBacktrack*************************/
/*****************************************************************/
void addWallBacktrack(maze_t* maze, XYPos* from, int dir){

    /* Defensive programing ********************************/
    // Null check
    if (maze == NULL || from == NULL){ 
        return;
    }

    // safety checks for from position
    if ((from->x > maze->width || from->y > maze->height) || (from->x < 0 || from->y < 0)) {
        return;
    }
    // saftey checks for direction
    if(dir < 0 || dir > 3){ return;}

    // Edge wall check. because we do not want to add edge walls ofc
    if(dir == 0 && from->x == 0){
        return;
    }
    else if(dir == 1 && from->y == 0){
        return;
    }
    else if(dir == 2 && from->y == maze->height - 1){
        return;
    }
    else if (dir == 3 && from->x == maze->width - 1){
        return;
    }

    /* Do the stuff programing ********************************/
    // Make the 'to' point
    XYPos* to = malloc(sizeof(XYPos));

    // Malloc check
    if(to == NULL){return;}

    if(dir == 0){
        to->x = from->x - 1; to->y = from->y;
    }
    else if(dir == 1){
        to->x = from->x; to->y = from->y - 1;
    }
    else if(dir == 2){
        to->x = from->x; to->y = from->y + 1;
    }
    else{
        to->x = from->x + 1; to->y = from->y;
    }

    // Add wall to matrix
    if((int)from->y == (int)to->y){
        if((int)from->x - (int)to->x > 0){
            // It is a verticle wall encountered from its right
            maze->matrix[(2 * (int)from->y)][1 + (2 * (int)to->x)] = -2;
        }
        else{
            // It is a verticle wall encountered from its left
            maze->matrix[(2 * (int)from->y)][1 + (2 * (int)from->x)] = -2;
        }
    }
    else{
        if((int)from->y - (int)to->y > 0){
            // it is a horizontal wall encountered from its bottom
            maze->matrix[1 + (2 * (int)to->y)][(2 * (int)from->x)] = -2;
        }
        else{
            // it is a horizontal wall encountered from its top
            maze->matrix[1 + (2 * (int)from->y)][(2 * (int)from->x)] = -2;
        }
    }

    free(to);
}

/****************************setAvatar****************************/
/*****************************************************************/
void setAvatar(maze_t* maze, char id, XYPos* loc){

    /* Defensive programing ********************************/
    // Null checks
    if (maze == NULL || loc == NULL){ 
        return;
    }

    /* Actual programing ***********************************/
    if(loc->x > maze->width || loc->y > maze->height){return;}          // Ensure that location is valid
    maze->GUI[1 + (2 * (int)loc->y)][(4 * ((int)loc->x + 1)) - 2] = id; // Set the character in the appropriate location

}

/****************************printMaze****************************/
/*****************************************************************/
void printMaze(maze_t* maze){

    /* Defensive programing ********************************/
    // Null checks
    if (maze == NULL){ 
        return;
    }

    /* Actual programing ***********************************/
    for(int i = 0; i < maze->height * 2 + 1; i++){                       // Loop through array of maze GUI lines
        fprintf(stdout, "%s\n", maze->GUI[i]);                           // Print each line
    } 
}

/****************************wallCheck****************************/
/*****************************************************************/
bool wallCheck(maze_t* maze, XYPos* from, int dir){

    /* Defensive programing ********************************/
    // Null checks
    if (maze == NULL || from == NULL){ 
        return true;
    }

    // Validate from position
    if(from->x > maze->width || from->y > maze->height){return true;}

    // Validate direction
    if(dir < 0 || dir > 3){ return true;}

    // Edge wall check
    if(dir == 0 && from->x == 0){
        return true;
    }
    else if(dir == 1 && from->y == 0){
        return true;
    }
    else if(dir == 2 && from->y == maze->height - 1){
        return true;
    }
    else if (dir == 3 && from->x == maze->width - 1){
        return true;
    }

    /* Actual programing ************************************/
    // Make the to point
    XYPos* to = malloc(sizeof(XYPos));
    if(dir == 0){
        to->x = from->x - 1; to->y = from->y;
        if(maze->matrix[(2 * (int)from->y)][1 + (2 * (int)to->x)] == -2){
            // It is a verticle wall encountered from its right
            free(to);
            return true;
        }
    }
    else if(dir == 1){
        to->x = from->x; to->y = from->y - 1;
        if(maze->matrix[1 + (2 * (int)to->y)][(2 * (int)from->x)] == -2){
            // it is a horizontal wall encountered from its bottom
            free(to);
            return true;
        }
    }
    else if(dir == 2){
        to->x = from->x; to->y = from->y + 1;
        if(maze->matrix[1 + (2 * (int)from->y)][(2 * (int)from->x)] == -2){
            // it is a horizontal wall encountered from its top
            free(to);
            return true;
        }
    }
    else{
        to->x = from->x + 1; to->y = from->y;
        if(maze->matrix[(2 * (int)from->y)][1 + (2 * (int)from->x)] == -2){
            // It is a verticle wall encountered from its left
            free(to);
            return true;
        }
    }

    free(to);
    return false;
}

/**************************deadEndCheck***************************/
/*****************************************************************/
int deadEndCheck(maze_t* maze, XYPos* loc){
    /* Defensive programing ********************************/
    // Null checks
    if (maze == NULL || loc == NULL){ 
        return -2;
    }

    /* Actual programing ********************************/
    int wallsFound = 0;
    int freeDirection = -8;
    
    //// CHECK ALL THE DIRECTIONS
    if(wallCheck(maze, loc, 0)){
        wallsFound++;
    }
    else{
        freeDirection = 0;
    }

    if(wallCheck(maze, loc, 1)){
        wallsFound++;
    }
    else{
        freeDirection = 1;
    }

    if(wallCheck(maze, loc, 2)){
        wallsFound++;
    }
    else{
        freeDirection = 2;
    }

    if(wallCheck(maze, loc, 3)){
        wallsFound++;
    }
    else{
        freeDirection = 3;
    }

    // Check if we are in a dead end
    if(wallsFound == 3){
        return freeDirection;
    }
    else{
        return -1;
    }
}

/****************************deleteMaze***************************/
/*****************************************************************/
bool deleteMaze(maze_t* maze){
    /* Defensive programing ********************************/
    if (maze == NULL){ return false;}
    else{
        // Free GUI
        for(int i = 0; i < maze->height * 2 + 1; i++){
            free(maze->GUI[i]);
        }
        free(maze->GUI);

        // Free matrix
        for(int i = 0; i < (maze->height * 2) - 1; i++){
            free(maze->matrix[i]); 
        }
        free(maze->matrix);

        // free it all
        free(maze);
    }
    return true;
}



