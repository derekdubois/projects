// Implements a dictionary's functionality

#include <ctype.h>
#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <stdlib.h>
#include <cs50.h>

#include "dictionary.h"

// Represents a node in a hash table
typedef struct node
{
    char word[LENGTH + 1];
    struct node *next;
}
node;

unsigned int wordcount;
unsigned int hashnumber;
// TODO: Choose number of buckets in hash table
const unsigned int N = 26;


// Hash table
node *table[N];

// Returns true if word is in dictionary, else false
bool check(const char *word)
{
    // TODO
    hashnumber = hash(word);
    node *holder = table[hashnumber];

    while(holder != 0)
    {
        if(strcasecmp(word, holder->word) == 0)
        {
            return true;
        }
        holder = holder->next;
    }
    return false;
}

// Hashes word to a number
unsigned int hash(const char *word)
{
    // TODO: Improve this hash function
    unsigned long sum = 0;
    for (int i = 0; i < strlen(word); i++)
    {
        sum += tolower(word[i]);
    }
    return sum % N;
}

// Loads dictionary into memory, returning true if successful, else false
bool load(const char *dictionary)
{
    //use fopen to open up dictionary file

    FILE *DictionaryFile = fopen(dictionary, "r");
    if (DictionaryFile == NULL)
    {
        return false;
    }

    //to read char *
    char word[LENGTH + 1];
    while(fscanf(DictionaryFile, "%s", word) != EOF)
    {
        //allocate memory for new nodes
        node *temp = malloc(sizeof(node));
        if(temp == NULL)
        {
            return false;
        }
        strcpy(temp->word, word);

        //implement hash
        hashnumber = hash(word);

        //check if points to NULL
        if(table[hashnumber] == NULL)
        {
            temp->next = NULL;
        }
        else
        {
            //point temp to 1st node in the LL
            temp->next = table[hashnumber];
        }
        table[hashnumber] = temp;

        wordcount += 1;
    }
    fclose(DictionaryFile);

    // TODO
    return true;
}

// Returns number of words in dictionary if loaded, else 0 if not yet loaded
unsigned int size(void)
{
    // TODO
    if (wordcount > 0)
    {
        return wordcount;
    }
    return 0;
}


// Unloads dictionary from memory, returning true if successful, else false
bool unload(void)
{
    // TODO
    for(int i = 0; i < N; i++)
    {
        node *holder = table[i];

        while(holder != NULL)
        {
            node *temp = holder;
            holder = holder->next;
            free(temp);
        }
        if (holder == NULL && i == N-1)
        {
            return true;
        }
    }
    return false;
}
