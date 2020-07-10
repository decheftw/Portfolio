/* 
 * toolkit struct - function called by each avatar thread
 *  nn
 */

#include <netdb.h>
#include "amazing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>


typedef struct maze maze_t;

/**************** checktype ****************/
/*
 * Receives a uint32_t type in network byte form. The network byte form
 * is converted into host byte form in the function.
 * 
 * usage example: if (checktype(response.type) == -1)
 * 
 * Returns integer based on message type:
 * -1 - error type (outputs error statement)
 * 0 - AM_INIT_OK type
 * 1 - AM_AVATARU_TURN
 * 2 - AM_MAZE_SOLVED
 * 3 - not an error nor any of the previous types (should never happen)
 */
int checktype(uint32_t type);

/****************************newMaze******************************/
/*
 * Creates and mallocs a new maze struct and returns it to the user
 * User is responsible for later freeing the struct using deleteMaze()
 */
maze_t* newMaze(int width, int height);

/****************************addWall******************************/
/*
 * Inserts a wall in the given maze struct's GUI and matrix
 * User defines the location of the wall by defining
 */
void addWall(maze_t* maze, XYPos* from, int dir);

/****************************setAvatar****************************/
/*
 * Inserts an avatar into the given maze struct's GUI
 * User defines the location by providing XYPos struct
 */
void setAvatar(maze_t* maze, char id, XYPos* loc);

/****************************printMaze****************************/
/*
 * Prints maze GUI
 * 
 */
void printMaze(maze_t* maze);

/****************************wallCheck****************************/
/*
 * Checks if a wall exists in between two positions in the maze. 
 * User provides the two positions
 * 
 * Returns true if wall exists and false if there is no wall
 */
bool wallCheck(maze_t* maze, XYPos* from, int dir);

/**************************deadEndCheck***************************/
/*
 * Check to see if a position is in a dead end
 * 
 * Returns:
 *      (-1 ) - not in a dead end
 *      (-2 ) - maze is null
 *      (-3 ) - oof this is bad
 *      ( x ) - position x is free 
 */
int deadEndCheck(maze_t* maze, XYPos* loc);

/************************addWallBacktrack*************************/
/*
 * Inserts a wall in the given maze struct's matrix and not GUI
 * User defines the location of the wall by defining
 */
void addWallBacktrack(maze_t* maze, XYPos* from, int dir);

/****************************deleteMaze***************************/
/*
 * Deletes and frees maze struct
 * Returns false if delete was unsuccesfull
 */
bool deleteMaze(maze_t* maze);

