/* 
 * counters.c - CS50 'counters' module
 *
 * see counters.h for more information.
 *
 * Gao Chen, 1/31/2020
 * Used bag.c as a scaffold
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "counters.h"
#include "memory.h"

/**************** file-local global variables ****************/
/* none */

/**************** local types ****************/
typedef struct countersnode {
  int key;  // pointer to data for this item
  int freq;
  struct countersnode *next;	      // link to next node
} countersnode_t;

/**************** global types ****************/
typedef struct counters {
  struct countersnode *head;	      // head of the list of items in bag
} counters_t;

/**************** global functions ****************/
/* that is, visible outside this file */
/* see counters.h for comments about exported functions */

/**************** local functions ****************/
/* not visible outside this file */
/**************** countersnode_new ****************/
/* Allocate and initialize a countersnode */
// the 'static' modifier means this function is not visible
// outside this file
static countersnode_t *countersnode_new(const int key)
{
  countersnode_t *node = count_malloc(sizeof(countersnode_t));

  if (node == NULL) {
  // error allocating memory for node; return error
     return NULL;
  } else {
	node->key = key;
	node->freq = 1;
	node->next = NULL;
	return node;
  }
}


/**************** counters_new() ****************/
counters_t* counters_new(void)
{
  counters_t *counters = count_malloc(sizeof(counters_t));

  if (counters == NULL) {
    return NULL; // error allocating counter
  } else {
    // initialize contents of counters structure
    counters->head = NULL;
    return counters;
  }
}

/**************** counters_add ****************/
int counters_add(counters_t *ctrs, const int key)
{
  printf("starting\n");
	if (ctrs != NULL || key < 0) {
	  for (countersnode_t *node = ctrs->head; node != NULL; node = node->next){
		  if (node->key == key){
			node->freq++;
			printf("done1\n");
			return node->freq;
		}
	  }
    			// allocate a new node to be added to the list
    			countersnode_t *new = countersnode_new(key);
			if (new != NULL) {
      				// add it to the head of the list
      				new->next = ctrs->head;
      				ctrs->head = new;
				printf("done2\n");
				return 1;
			}
    }
  return 0;
}


/**************** counters_get() ****************/
int counters_get(counters_t *ctrs, const int key)
{
  if (ctrs == NULL) {
    return 0; // bad counter
  } else if (ctrs->head == NULL) {
    return 0; // counter is empty
  } else {
    for (countersnode_t *node = ctrs->head; node != NULL; node = node->next){
	    if (node->key == key){
		    return node->freq;
	    }
	}
  }
  return 0;
}

/**************** counters_set() ****************/
bool counters_set(counters_t *ctrs, const int key, const int count)
{
  if (ctrs == NULL) {
    return false; // bad counter
  } else if (ctrs->head == NULL) {
    return false; // counter is empty
  } else {
    for (countersnode_t *node = ctrs->head; node != NULL; node = node->next){
            if (node->key == key){
                    node->freq = count;
            	    return true;
	    }
        }
  }
  return false;
}


/**************** counters_print() ****************/
void counters_print(counters_t *ctrs, FILE *fp)
{
  if (fp != NULL) {
    if (ctrs != NULL) {
      fputs("{ ", fp);
      for (countersnode_t *node = ctrs->head; node != NULL; node = node->next) {
        // print this node
          fputc('(', fp);
          fprintf(fp, "%d", node->key);
	  fputc(',', fp);
	  fprintf(fp, "%d", node->freq);
	  fputc(')', fp); 
        }
      }
      fputs(" }\n", fp);
    } else {
      fputs("(null)", fp);
    }
  }


/**************** counters_iterate() ****************/
void counters_iterate(counters_t *ctrs, void *arg, void (*itemfunc)(void *arg, const int key, const int count) )
{
  if (ctrs != NULL && itemfunc != NULL) {
    // call itemfunc with arg, on each item
    for (countersnode_t *node = ctrs->head; node != NULL; node = node->next) {
      (*itemfunc)(arg, node->key, node->freq); 
    }
  }
}

/**************** counters_delete() ****************/
void counters_delete(counters_t *ctrs)
{
  if (ctrs != NULL) {
    for (countersnode_t *node = ctrs->head; node != NULL; ) {
      	countersnode_t *next = node->next;	    // remember what comes next
      	count_free(node);			    // free the node
      	node = next;			    // and move on to next
    }

    count_free(ctrs);
  }

#ifdef MEMTEST
  count_report(stdout, "End of bag_delete");
#endif
}
