#include "toolkit.h"

/**************** main() ****************/
int main(const int argc, char *argv[]){

    maze_t* maze = newMaze(10, 10);


    XYPos* from = malloc(sizeof(XYPos));
    
    from->x = 3; from->y = 3;
    
    addWall(maze, from, 0);
    addWall(maze, from, 1);
    addWall(maze, from, 2);
    addWall(maze, from, 3);
    setAvatar(maze, 'c', from);
    printMaze(maze);

    if(wallCheck(maze, from, 0)){
        printf("There is a wall\n");
    }else{printf("No wall\n");}

    if(wallCheck(maze, from, 1)){
        printf("There is a wall\n");
    }else{printf("No wall\n");}

    if(wallCheck(maze, from, 2)){
        printf("There is a wall\n");
    }else{printf("No wall\n");}

    if(wallCheck(maze, from, 3)){
        printf("There is a wall\n");
    }else{printf("No wall\n");}

    // boundry wall tests
    setAvatar(maze, ' ', from);
    from->x = 0; from->y = 0;

    setAvatar(maze, 'c', from);
    printMaze(maze);

    if(wallCheck(maze, from, 0)){
        printf("There is a wall\n");
    }else{printf("No wall\n");}

    if(wallCheck(maze, from, 1)){
        printf("There is a wall\n");
    }else{printf("No wall\n");}

    if(wallCheck(maze, from, 2)){
        printf("There is a wall\n");
    }else{printf("No wall\n");}

    if(wallCheck(maze, from, 3)){
        printf("There is a wall\n");
    }else{printf("No wall\n");}

    setAvatar(maze, ' ', from);
    from->x = 9; from->y = 9;

    setAvatar(maze, 'c', from);
    printMaze(maze);

    if(wallCheck(maze, from, 0)){
        printf("There is a wall\n");
    }else{printf("No wall\n");}

    if(wallCheck(maze, from, 1)){
        printf("There is a wall\n");
    }else{printf("No wall\n");}

    if(wallCheck(maze, from, 2)){
        printf("There is a wall\n");
    }else{printf("No wall\n");}

    if(wallCheck(maze, from, 3)){
        printf("There is a wall\n");
    }else{printf("No wall\n");}


    free(from);

    deleteMaze(maze);
    return 0;
}