/* 
 * set.c - CS50 'set' module
 *
 * see set.h for more information.
 *
 * Katherine Taylor, Jan. 2020
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "set.h"
#include "memory.h"


/********************* file-local global variables **************/
/* none */

/*************** local types ******************/
typedef struct setnode {
	void *item;					// pointer to its data
	struct setnode *next;		// link to next node
	char *key;
} setnode_t;


/*************** global types *******************/
typedef struct set {
	struct setnode *head;		// head of the list of set nodes
} set_t;


/*************** local functions ******************/
//static setnode_t *setnode_new(void *item);


/*** set_new ***/
set_t* set_new(void)
{
	set_t *set = count_malloc(sizeof(set_t));

	if (set == NULL) {
		return NULL;
	} else {
		set->head = NULL;
		return set;
	}
}

/*** set_insert ***/
bool set_insert(set_t *set, const char *key, void *item)
{
	if ((set != NULL) && (item != NULL) && (key != NULL)) {						// if all inputs are valid
		 for (setnode_t *node = set->head; node != NULL; node = node->next) {	// loop through linked list
			 if (! strcmp(node->key, key)) {									// if the key already exists in the list
				 return false;
			 }
		 }
		setnode_t *new = count_malloc(sizeof(setnode_t));
		char *newKey = count_malloc(sizeof(key));								// creating a copy of the key
		strcpy(newKey, key);
	
		// setting values of the new node
		new->item = item;
		new->key = newKey;
		new->next = set->head;
		set->head = new;
		return true;
	}
	else {
		return false;
	}
}

/*** set_find ***/
void *set_find(set_t *set, const char *key) {
	if ((set == NULL) || (key == NULL)) {		// testing for valid inputs
		return NULL;
	}
	else if (set->head == NULL) {				// testing that the set is not empty
		return NULL;
	}
	else {
		for (setnode_t *node = set->head; node != NULL; node = node->next) {
			if (! strcmp(node->key, key)) {		// if the keys are the same
				return node;
			}
		}
		return NULL;
	}
}


/*** set_print ***/
void set_print(set_t *set, FILE *fp, void (*itemprint)(FILE *fp, const char *key, void *item) ) {
	if (fp != NULL) {
		if (set != NULL) {
			fputc('{', fp);
			for (setnode_t *node = set->head; node != NULL; node = node->next) {
				if (itemprint != NULL) {
					// calling given itemprint function
					fputc(' ', fp);
					(*itemprint)(fp, node->key, node->item);
					fputc(',', fp);
				}
			}
			fputs(" }\n", fp);
		}
		else {
			fputs("(null)", fp);
		}
	}
}


/*** set_iterate ***/
void set_iterate(set_t *set, void *arg, void (*itemfunc)(void *arg, const char *key, void *item) ) {
	if (set != NULL && itemfunc != NULL) {
		// call itemfunc with arg on each item
		for (setnode_t *node = set->head; node != NULL; node = node->next) {
			(*itemfunc)(arg, node->key, node->item);
		}
	}
}


/*** set_delete ***/
void set_delete(set_t *set, void (*itemdelete)(void *item) ) {
	if (set != NULL) {
		for (setnode_t *node = set->head; node != NULL; ) {
			if (itemdelete != NULL) {
				(*itemdelete)(node->item);
			}
			setnode_t *next = node->next;
			if (node->key != NULL) {
				count_free(node->key);
			}
			count_free(node);
			node = next;
		}
		count_free(set);
	}
	#ifdef MEMTEST
		count_report(stdout, "end of bag_delete");
	#endif
}
