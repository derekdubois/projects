#include <ctype.h>
#include <cs50.h>
#include <stdio.h>
#include <string.h>

// Points assigned to each letter of the alphabet
int POINTS[] = {1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10};

int compute_score(string word);

int main(void)
{
    // Get input words from both players
    string word1 = get_string("Player 1: ");
    string word2 = get_string("Player 2: ");

    // Score both words
    int score1 = compute_score(word1);
    int score2 = compute_score(word2);

    // Print the winner
    if (score1 > score2)
    {
        printf("Player 1 wins!\n");
    }
    if (score2 > score1)
    {
        printf("Player 2 wins!\n");
    }
    if (score1 == score2)
    {
        printf("Tie!\n");
    }
}

int compute_score(string word)
{
    // Compute and return score for string
    //string output = input;
    int score = 0;

    for (int i = 0; i < strlen(word); i++)
    {
        char c = tolower(word[i]);

        switch (c)
        {
            case 'a':
                word[i] = 1;
                break;
            case 'b':
                word[i] = 3;
                break;
            case 'c':
                word[i] = 3;
                break;
            case 'd':
                word[i] = 2;
                break;
            case 'e':
                word[i] = 1;
                break;
            case 'f':
                word[i] = 4;
                break;
            case 'g':
                word[i] = 2;
                break;
            case 'h':
                word[i] = 4;
                break;
            case 'i':
                word[i] = 1;
                break;
            case 'j':
                word[i] = 8;
                break;
            case 'k':
                word[i] = 5;
                break;
            case 'l':
                word[i] = 1;
                break;
            case 'm':
                word[i] = 3;
                break;
            case 'n':
                word[i] = 1;
                break;
            case 'o':
                word[i] = 1;
                break;
            case 'p':
                word[i] = 3;
                break;
            case 'q':
                word[i] = 10;
                break;
            case 'r':
                word[i] = 1;
                break;
            case 's':
                word[i] = 1;
                break;
            case 't':
                word[i] = 1;
                break;
            case 'u':
                word[i] = 1;
                break;
            case 'v':
                word[i] = 4;
                break;
            case 'w':
                word[i] = 4;
                break;
            case 'x':
                word[i] = 8;
                break;
            case 'y':
                word[i] = 4;
                break;
            case 'z':
                word[i] = 10;
                break;
            default:
                word[i] = 0;
                break;
        }
        score = score + word[i];
    }
    return score;
}
