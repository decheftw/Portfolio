// This program connects to a server
// Made for group Seg Fault 
//  
#include "AMStartup.h"

// /************** global variables **************/
maze_t *maze; // maze struct
pthread_mutex_t lock;
int spacesLeft;
XYPos* loc;

/**************** main() ****************/
int main(const int argc, char *argv[])
{
  char *program;    // this program's name
  int port;         // server port
  int nAvatars;
  int difficulty;
  char *hostname;   // server hostname

  program = argv[0];
  if (argc != 4) {
    fprintf(stderr, "usage: %s numberavatars difficulty hostname\n", program);
    exit(1);
  }
  
  // initialize variables to command-line arguments
  port = atoi(AM_SERVER_PORT);   
  nAvatars = atoi(argv[1]);
  difficulty = atoi(argv[2]);
  hostname = argv[3];
  
  // validate arguments
  if (nAvatars > AM_MAX_AVATAR || nAvatars < 2) {
    fprintf(stderr, "number of avatars exceeds 10 (maximum) or is below the minimum of 2\n");
    exit(2);
  }
  if (difficulty > AM_MAX_DIFFICULTY) {
    fprintf(stderr, "difficulty exceeds 9 (maximum)\n");
    exit(3);
  }

  int comm_sock = socket(AF_INET, SOCK_STREAM, 0);
  if (comm_sock < 0) {
    perror("opening socket");
    exit(4);
  }
  
  // 2. Initialize the fields of the server address
  struct sockaddr_in server;  // address of the server
  server.sin_family = AF_INET;
  server.sin_port = htons(port);
  // Look up the hostname specified on command line
  struct hostent *hostp = gethostbyname(hostname); // server hostname
  if (hostp == NULL) {
    fprintf(stderr, "%s: unknown host '%s'\n", program, hostname);
    exit(5);
  }  
  memcpy(&server.sin_addr, hostp->h_addr_list[0], hostp->h_length);

  // 3. Connect the socket to that server   
  if (connect(comm_sock, (struct sockaddr *) &server, sizeof(server)) < 0) {
    perror("connecting stream socket");
    exit(6);
  }
  printf("Connected!\n");
  
  // initialize message sent to server
  AM_Message message;
  memset(&message, 0, sizeof(message)); // clear up the buffer
  message.init.nAvatars = htonl(nAvatars);
  message.init.Difficulty = htonl(difficulty);
  message.type = htonl(AM_INIT);

  if (write(comm_sock, &message, sizeof(message)) < 0) { // send AM_INIT message to server
    perror("writing on stream socket");
    exit(7);
  }

  AM_Message response; // response received from server
  if (read(comm_sock, &response, sizeof(response)) < 0) { // read AM_INIT_OK message in from server
    perror("reading on stream socket");
    exit(8);
  }
  if (checktype(response.type) == -1) { // error message received
    exit(9);    
  }

  int MazePort = ntohl(response.init_ok.MazePort);
  int MazeHeight = ntohl(response.init_ok.MazeHeight);
  int MazeWidth = ntohl(response.init_ok.MazeWidth);

  // initialize "global" variables
  maze_t *maze = newMaze(MazeWidth, MazeHeight);
  spacesLeft = MazeHeight * MazeWidth;
  loc = malloc(sizeof(XYPos));
  loc->x = 0;
  loc->y = 0;
  
  // create log file
  char filename[] = "Amazing_";
  // append USER_ to filename
  char* user = getenv("USER");
  strcat(filename, user);
  strcat(filename, "_");
  // append N_ to filename
  char n[5];
  sprintf(n, "%d", nAvatars);
  strcat(filename, n);
  strcat(filename, "_");
  // append Difficulty_ to filename
  char d[5];
  sprintf(d, "%d", difficulty);
  strcat(filename, d);
  // append .log to filename
  strcat(filename, ".log");

  // write into file
  FILE* fp = fopen(filename, "w");
  time_t curtime;
  time(&curtime);
  fprintf(fp, "%s\t%d\t%s\n", user, MazePort, ctime(&curtime));
  fclose(fp);

  initialize_threads(nAvatars, difficulty, hostname, MazePort, filename, maze);

  // Clean up
  free(loc);                                                 // Freeeeeeeeeee
  pthread_mutex_destroy(&lock); 
  deleteMaze(maze);                                          // bye bye maze
}

void initialize_threads(int nAvatars, int difficulty,char * hostname, int mazePort, char* filename, maze_t* maze)
{
  pthread_t t1[nAvatars];
  if(pthread_mutex_init(&lock, NULL) != 0) { 
    printf("\n mutex init has failed\n");
  }

  for (int i = 0; i < nAvatars; i++) {
    // Set the globally shared data
    avStatus_t* stat = avStatus_new(); 
    avStatus_setAvatarId(stat, i);
    avStatus_setnAvatars(stat, nAvatars);
    avStatus_setdifficulty(stat, difficulty);   // initializes struct members
    avStatus_sethostname(stat, hostname);
    avStatus_setmazePort(stat, mazePort);
    avStatus_setfilename(stat, filename);
    avStatus_setmaze(stat, maze);
    avStatus_setfinalPoint(stat, loc);
    avStatus_setspacesLeft(stat, &spacesLeft);
    avStatus_setlock(stat, &lock);

    pthread_create(&t1[i], NULL, run_avatar, (void*)stat);  // Create threads
  }
  for (int i = 0; i <nAvatars; i++) {
    pthread_join(t1[i], NULL);
  }
}
