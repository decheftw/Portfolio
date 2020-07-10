/* 
 * avatar functions - function called by each avatar thread
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>       // read, write, close
#include <string.h>       // memcpy, memset
#include <netdb.h>        // socket-related structures
#include <pthread.h>
#include <time.h>
#include "amazing.h"
#include "avatar.h"
#include "toolkit.h"
#include "./libcs50/hashtable.h"

/******************************* initialize_threads ********************************/
/*
 * takes all the members of the avStatus struct and initializes it for each thread
 */
void initialize_threads(int nAvatars, int difficulty,char * hostname, int mazePort, char* filename, maze_t* maze);
