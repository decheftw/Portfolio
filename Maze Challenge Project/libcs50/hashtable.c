/* 
 * hashtable.c - CS50 'hashtable' module
 *
 * see hashtable.h for more information.
 *
 * Gao Chen, 2/1/2020
 * Used bag.c as a scaffold
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hashtable.h"
#include "memory.h"
#include "../set/set.h"
#include "jhash.h"
#include "math.h"

/**************** file-local global variables ****************/
/* none */

/**************** local types ****************/

/**************** global types ****************/
typedef struct hashtable {
  struct set **array;	      // array of pointers to sets
  int num_slots;
} hashtable_t;

/**************** global functions ****************/
/* that is, visible outside this file */
/* see bag.h for comments about exported functions */

/**************** local functions ****************/
/* not visible outside this file */

/**************** hashtable_new() ****************/
hashtable_t *hashtable_new(const int num_slots)
{	
  hashtable_t *ht = count_malloc(sizeof(hashtable_t));
  if (ht == NULL) {
    return NULL; // error allocating hashtable
  } else {
    // initialize contents of hashtable structure
    ht->array = calloc(num_slots + 1, sizeof(set_t*) + 1);
    ht->num_slots = num_slots;
    for (int i = 0; i < num_slots; i++) {
	    ht->array[i] = set_new();
  }
  }
    return ht;

}
/**************** hashtable_insert() ****************/
bool hashtable_insert(hashtable_t *ht, const char *key, void *item)
{
  if (ht != NULL && item != NULL && key != NULL) {
    int i = JenkinsHash(key, ht->num_slots);
	if (set_find(ht->array[i], key) != NULL){
	  return false;
	  }
      set_insert(ht->array[i], key, item);
      return true;
    }
  

#ifdef MEMTEST
  count_report(stdout, "After bag_insert");
#endif
  return false;
}


/**************** hashtable_find() ****************/
void* hashtable_find(hashtable_t *ht, const char *key)
{
  if (ht == NULL || key == NULL) {
    return NULL; // bad ht or key
  } else {
	int i = JenkinsHash(key, 3);
	return set_find(ht->array[i], key);
  }
    return NULL;
}
/**************** hashtable_print() ****************/
void hashtable_print(hashtable_t *ht, FILE *fp, void (*itemprint)(FILE *fp, const char *key, void *item) )
{
  if (fp != NULL) {
	if (ht != NULL) {
      		for (int i = 1; i <= ht->num_slots; i++){
			fprintf(fp, "%d.", i);
      			set_print(ht->array[i], fp, itemprint);
        	
      }
      fputs(" }\n", fp);
	}
	else {
      fputs("(null)", fp);
    }
  }
}
/**************** hashtable_iterate() ****************/
void hashtable_iterate(hashtable_t *ht, void *arg, void (*itemfunc)(void *arg, const char *key, void *item) )
{
  if (ht != NULL && itemfunc != NULL) {
    // call itemfunc with arg, on each item
    for (int i = 0; i < ht->num_slots; i++){
	set_iterate(ht->array[i], arg, itemfunc);	  
    
   }
  }
}

/**************** hashtable_delete() ****************/
void hashtable_delete(hashtable_t *ht, void (*itemdelete)(void *item) )
{
  printf("deleting!!!\n");
  if (ht != NULL) {
    for (int i = 0; i < ht->num_slots; i++) {
	    if (itemdelete != NULL) {		    // if possible...
        set_delete(ht->array[i], *itemdelete);
		// calling set_delete from set.h item
      }
     }
    count_free(ht->array);
    count_free(ht);
  }

#ifdef MEMTEST
  count_report(stdout, "End of bag_delete");
#endif
}
