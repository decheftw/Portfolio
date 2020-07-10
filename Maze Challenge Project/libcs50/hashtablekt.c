/* 
 * hashtable.c - CS50 'hashtable' module
 *
 * see hashtable.h for more information.
 *
 * Katherine Taylor, Jan. 2020
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../set/set.h"
#include "../set/set.c"
#include "memory.h"
#include "hashtable.h"
#include "jhash.c"
#include "jhash.h"


/**************** global types ****************/
typedef struct hashtable {
    int size;
    struct set **table;     // set of sets
} hashtable_t;


/**************** functions ****************/

/*** hashtable_new ***/
hashtable_t *hashtable_new(const int num_slots) {
    if (num_slots > 0) {
        hashtable_t *ht = count_malloc(sizeof(hashtable_t));

        set_t **set = calloc(num_slots, sizeof(set_t*));

        ht->table = set;
        ht->size = num_slots;
        return ht;
    }
    else {
        return NULL;
    }
}

/*** hashtable_insert ***/
bool hashtable_insert(hashtable_t *ht, const char *key, void *item) {
    if ((ht != NULL) && (key != NULL) && (item != NULL)) {
        unsigned long index = JenkinsHash(key, ht->size);

        if (ht->table[index] == NULL) {                             // if the index doesn't already exist
            set_t *new = set_new();
            ht->table[index] = new;
            if (set_insert(new, key, item)) {
                return true;
            }
            else {
                return false;
            }
        }
        else {                                                      // if the index already exists
            set_t *set = ht->table[index];
            if (set_insert(set, key, item)) {                     // if the key is not already in the set at the index
                return true;
            }
            else {                                                  // if the key already exists
                return false;
            } 
        }
    }
    return false;
}

/*** hashtable_find ***/
void *hashtable_find(hashtable_t *ht, const char *key) {
    if ((ht != NULL) && (key != NULL)) {
        unsigned long index = JenkinsHash(key, ht->size);

        if (ht->table[index] != NULL) {
            return set_find(ht->table[index], key);
        }
    }
    return NULL;
}



/*** hashtable_print ***/
void hashtable_print(hashtable_t *ht, FILE *fp, void (*itemprint)(FILE *fp, const char *key, void *item)) {
    if (fp != NULL) {
        if (ht != NULL) {
            if (itemprint != NULL) {
                for (int i = 0; i < ht->size; i++) {
                    fprintf(fp, "%d:", i);                  // the hash index
                    if (ht->table[i] == NULL) {
                       fputs("\n", fp);                     // leave blank if null
                    }
                    else {
                        set_print(ht->table[i], fp, itemprint);
                    }
                }
            }
        }
        else {
            fputs("(null)", fp); 
        }
    }
}


/*** hashtable_iterate ***/
void hashtable_iterate(hashtable_t *ht, void *arg, void (*itemfunc)(void *arg, const char *key, void *item)) {
    if ((ht != NULL) && (itemfunc != NULL)) {
        for (int i = 0; i < ht->size; i++) {
            if (ht->table[i] != NULL) {
                set_iterate(ht->table[i], arg, itemfunc);
            }
        }
    }
}



/*** hashtable_delete ***/
void hashtable_delete(hashtable_t *ht, void (*itemdelete)(void *item) ) {
    if ((ht != NULL) && (itemdelete != NULL)) {
        for (int i = 0; i < ht->size; i++) {
            if (ht->table[i] != NULL) {
                set_delete(ht->table[i], itemdelete);
            }
        }
        if (ht->table != NULL) {
            count_free(ht->table);
        }
        count_free(ht);
    }
}