/* 
 * set.c - CS50 'set' module
 *
 * see set.h for more information.
 *
 * Gao Chen, 1/28/2020
 * Used bag.c as a scaffold
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "set.h"
#include "memory.h"

/**************** file-local global variables ****************/
/* none */

/**************** local types ****************/
typedef struct setnode {
  char *key;			// pointer to key for this item
  void *item;		      // pointer to data for this item
  struct setnode *next;	      // link to next node
} setnode_t; 

/**************** global types ****************/
typedef struct set {
  struct setnode *head;	      // head of the list of items in set
} set_t;

/**************** global functions ****************/
/* that is, visible outside this file */
/* see bag.h for comments about exported functions */

/**************** local functions ****************/
/* not visible outside this file */
static setnode_t *setnode_new(const char *key, void *item);

/**************** set_new() ****************/
set_t* set_new(void)
{
  set_t *set = count_malloc(sizeof(set_t));

  if (set == NULL) {
    return NULL; // error allocating set
  } else {
    // initialize contents of set structure
    set->head = NULL;
    return set;
  }
}

/**************** set_insert() ****************/
bool set_insert(set_t *set, const char *key, void *item)
{
  for (setnode_t *node = set->head; node != NULL; node = node->next) {
	  if (key == node->key){
		  return false;
	  }
  }
  if (set != NULL && item != NULL && key != NULL) {
    // allocate a new node to be added to the list
    setnode_t *new = setnode_new(key, item);
    if (new != NULL) {
      // add it to the head of the list
      new->next = set->head;
      set->head = new;
      return true;
    }
  }


#ifdef MEMTEST
  count_report(stdout, "After bag_insert");
#endif
return false;
}


/**************** setnode_new ****************/
/* Allocate and initialize a setnode */
// the 'static' modifier means this function is not visible 
// outside this file
static setnode_t* setnode_new(const char *key, void *item)
{
  setnode_t *node = count_malloc(sizeof(setnode_t));
  node->key = malloc(sizeof(setnode_t));
  if (node == NULL) {
    // error allocating memory for node; return error
    return NULL;
  } else {
    strcpy(node->key, key);	  
    node->item = item;
    node->next = NULL;
    return node;
  }
}

/**************** set_find() ****************/
void* set_find(set_t *set, const char *key)
{
  if (set == NULL || key == NULL) {
    return NULL; // bad set or key
  } else if (set->head == NULL) {
    return NULL; // set is empty
  } else {
    for (setnode_t *node = set->head; node != NULL; node = node->next) {
    	if (key == node->key){
	return node->item;
  	}
    }
    return NULL;
}
}

/**************** set_print() ****************/
void set_print(set_t *set, FILE *fp, void (*itemprint)(FILE *fp, const char *key, void *item) )
{
  if (fp != NULL) {
    if (set != NULL) {
      fputc('{', fp);
      for (setnode_t *node = set->head; node != NULL; node = node->next) {
        // print this node
        if (itemprint != NULL) {  // print the node's item 
          fputc(' ', fp);
          (*itemprint)(fp, node->key, node->item); 
        }
      }
      fputs(" }\n", fp);
    } else {
      fputs("(null)", fp);
    }
  }
}

/**************** set_iterate() ****************/
void set_iterate(set_t *set, void *arg, void (*itemfunc)(void *arg, const char *key, void *item) )
{
  if (set != NULL && itemfunc != NULL) {
    // call itemfunc with arg, on each item
    for (setnode_t *node = set->head; node != NULL; node = node->next) {
      (*itemfunc)(arg, node->key, node->item); 
    }
  }
}

/**************** set_delete() ****************/
void set_delete(set_t *set, void (*itemdelete)(void *item) )
{
  if (set != NULL) {
    for (setnode_t *node = set->head; node != NULL; ) {
      if (itemdelete != NULL) {		    // if possible...
        (*itemdelete)(node->key);
      }
      setnode_t *next = node->next;	    // remember what comes next
      count_free(node);			    // free the node
      node = next;			    // and move on to next
    }

    count_free(set);
  }

#ifdef MEMTEST
  count_report(stdout, "End of bag_delete");
#endif
}
