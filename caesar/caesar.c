#include <cs50.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

bool just_digits(string text);


int main(int argc, string argv[])
{

    char upper_alpha[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
    char lower_alpha[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
    char final_text[] = "";
    int cypher;


    if(argc != 2 || !just_digits(argv[1]))
    {
        printf("Usage: ./caesar key\n");
        return 1;
    }
    string plain_text = get_string("plaintext: ");
    cypher = atoi(argv[1]);

    printf("ciphertext: ");

    for (int i = 0; i < strlen(plain_text); i++)
    {
        char c = plain_text[i];
        int position;
        char new_letter;

        switch (c)
        {
            case 'a':
                position = 0;
                break;
            case 'A':
                position = 0;
                break;
            case 'b':
                position = 1;
                break;
            case 'B':
                position = 1;
                break;
            case 'c':
                position = 2;
                break;
            case 'C':
                position = 2;
                break;
            case 'd':
                position = 3;
                break;
            case 'D':
                position = 3;
                break;
            case 'e':
                position = 4;
                break;
            case 'E':
                position = 4;
                break;
            case 'f':
                position = 5;
                break;
            case 'F':
                position = 5;
                break;
            case 'g':
                position = 6;
                break;
            case 'G':
                position = 6;
                break;
            case 'h':
                position = 7;
                break;
            case 'H':
                position = 7;
                break;
            case 'i':
                position = 8;
                break;
            case 'I':
                position = 8;
                break;
            case 'j':
                position = 9;
                break;
            case 'J':
                position = 9;
                break;
            case 'k':
                position = 10;
                break;
            case 'K':
                position = 10;
                break;
            case 'l':
                position = 11;
                break;
            case 'L':
                position = 11;
                break;
            case 'm':
                position = 12;
                break;
            case 'M':
                position = 12;
                break;
            case 'n':
                position = 13;
                break;
            case 'N':
                position = 13;
                break;
            case 'o':
                position = 14;
                break;
            case 'O':
                position = 14;
                break;
            case 'p':
                position = 15;
                break;
            case 'P':
                position = 15;
                break;
            case 'q':
                position = 16;
                break;
            case 'Q':
                position = 16;
                break;
            case 'r':
                position = 17;
                break;
            case 'R':
                position = 17;
                break;
            case 's':
                position = 18;
                break;
            case 'S':
                position = 18;
                break;
            case 't':
                position = 19;
                break;
            case 'T':
                position = 19;
                break;
            case 'u':
                position = 20;
                break;
            case 'U':
                position = 20;
                break;
            case 'v':
                position = 21;
                break;
            case 'V':
                position = 21;
                break;
            case 'w':
                position = 22;
                break;
            case 'W':
                position = 22;
                break;
            case 'x':
                position = 23;
                break;
            case 'X':
                position = 23;
                break;
            case 'y':
                position = 24;
                break;
            case 'Y':
                position = 24;
                break;
            case 'z':
                position = 25;
                break;
            case 'Z':
                position = 25;
                break;
        }
        if (isupper(c))
        {
            new_letter = upper_alpha[(position + cypher) % 26];
        }
        if (islower(c))
        {
            new_letter = lower_alpha[(position + cypher) % 26];
        }
        if (isspace(c))
        {
            new_letter = ' ';
        }
        if (!isalpha(c))
        {
            new_letter = c; 
        }
        printf("%c", new_letter);
    }
    printf("\n");
    //printf("%c\n", final_text);
    return 0;
}


bool just_digits(string text)
{
    for (int i = 0; i < strlen(text); i++)
    {
        if (!isdigit(text[i]))
        {
            return false;
        }
    }
    return true;
}