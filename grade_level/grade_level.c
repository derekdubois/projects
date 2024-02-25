#include <cs50.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

int count_letters(string user_input);
int count_words(string user_input);
int count_sentences(string user_input);

int main(void)
{

    string user_input = get_string("Text: ");
    int total_letters = count_letters(user_input);
    int total_words = count_words(user_input);
    int total_sentences = count_sentences(user_input);

    float letters_average = ((float) total_letters / (float) total_words) * 100;
    float sentences_average = ((float) total_sentences / (float) total_words) * 100;

    int grade_level = round(0.0588 * letters_average - 0.296 * sentences_average - 15.8);

    if (grade_level < 1)
    {
        printf("Before Grade 1\n");
    }
    else if (grade_level >= 16)
    {
        printf("Grade 16+\n");
    }
    else
    {
        printf("Grade %i\n", grade_level);
    }

}



int count_letters(string user_input)
{
    int letters = 0;

    for (int i = 0; i < strlen(user_input); i++)
    {
        if (isalpha(user_input[i]))
        {
            letters++;
        }
    }
    return letters;
}

int count_words(string user_input)
{
    int words = 1;

    for (int i = 0; i < strlen(user_input); i++)
    {
        if (user_input[i] == ' ')
        {
            words++;
        }
    }
    return words;
}

int count_sentences(string user_input)
{
    int sentences = 0;

    for (int i = 0; i < strlen(user_input); i++)
    {
        if (user_input[i] == '.' || user_input[i] == '?' || user_input[i] == '!')
        {
            sentences++;
        }
    }
    return sentences;
}


