/* 
 * avatar functions - function called by each avatar thread
 *   
 */

#include "toolkit.h"
#include "amazing.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <netdb.h>
#include <unistd.h>       // read, write, close
#include "./libcs50/hashtable.h"
#include <pthread.h>


typedef struct avStatus avStatus_t;

avStatus_t* avStatus_new();

/********************* avStatus_setAvatarId ********************/
/*
 * takes an avStatus struct and an int
 * initializes the thread's id
 */
void avStatus_setAvatarId(avStatus_t* stat, int id);

/******************** avStatus_setnAvatars *********************/
/*
 * takes an avStatus struct and an int
 * initializes the number of avatars
 */
void avStatus_setnAvatars(avStatus_t* stat, int nAvatars);

/********************* avStatus_setdifficulty *******************/
/*
 * takes an avStatus struct and an int
 * initializes the difficulty for each thread
 */
void avStatus_setdifficulty(avStatus_t* stat, int difficulty);

/********************* avStatus_hostname ************************/
/*
 * takes an avStatus struct and a character pointer
 * initializes the hostname to connect to
 */
void avStatus_sethostname(avStatus_t* stat, char* hostname);

/******************** avStatus_setmazePort **********************/
/*
 * takes an avStatus struct and an int
 * initializes the mazePort that the thread connects to
 */
void avStatus_setmazePort(avStatus_t* stat, int mazePort);

/******************** avStatus_setfilename **********************/
/*
 * takes an avStatus struct and a character pointer
 * initializes the filename where the logfile will be written
 */
void avStatus_setfilename(avStatus_t* stat, char* filename);

/********************** avStatus_setmaze ************************/
/*
 * takes an avStatus struct and a maze struct
 * initializes the maze to store the walls
 */
void avStatus_setmaze(avStatus_t* stat, maze_t* maze);

/********************* avStatus_setspacesLeft *******************/
/*
 * takes an avStatus struct and an int
 * sets the amount of spaces left in the maze to be blocked off
 */
void avStatus_setspacesLeft(avStatus_t* stat, int* spacesLeft);

/********************** avStatus_setlock ***********************/
/*
 * takes an avStatus struct and a mutex lock
 * initializes the mutex lock
 */
void avStatus_setlock(avStatus_t* stat, pthread_mutex_t* lock);

/********************** avStatus_setmazePort *******************/
/*
 * takes an avStatus struct and an int
 * initializes the mazePort that the thread connects to
 */
void avStatus_delete(avStatus_t* stat);

/*************************** run_avatar *************************/
/* 
 * takes an avStatus struct and contains the whole maze solving algorithm.
 */
void* run_avatar(void* av1);

/****************************** move *****************************/
/*
 * function that attempts to move avatar in direction entered it is facing
 * takes a direction, an avStatus struct, and a socket to communicate to
 */
void move(int dir, avStatus_t* av, int comm_sock);

/****************************** getRight *************************/
/*
 * takes a direction and returns a direction one rotation clockwise
 */
int getRight(int dir);

/*************************** updatePosition ***********************/
// reads in a server message and updates the avatars current position
void updatePosition(avStatus_t* av, int comm_sock);

/*************************** backTrackme ***************************/
/*
 * It takes an avatar that has identified that it is stuck in a deadend and 
 * then moves it towards the open direction
 */
void backTrackme(avStatus_t* av, int comm_sock);

/************************** oppositeDirection ***********************/
/*
 * takes a direction and returns the opposite direction 
 */
int oppositeDirection(int dir);

/****************************** getLeft ****************************/
/*
 * takes a direction and returns the direction one rotation counter-clockwise 
 */
int getLeft(int dir);

/******************************* logging ***************************/
/*
 * takes an avStatus struct and a string
 * appends a string into a file stored in avStatus
 */
void logging(avStatus_t *av, const char *str);

/****************************rightHandRule**************************/
/*
 * Function that the "Right Hand Rule Heuristic on an avatar
 * takes in an avStruct_t
 * 
 */
void rightHandRule(avStatus_t* av, int comm_sock);

/******************************* logging ***************************/
/*
 * Sets the pointer to the global final point in the avStatus structs
 */
void avStatus_setfinalPoint(avStatus_t* stat, XYPos* loc);

