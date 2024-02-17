
#include <cs50.h>
#include <stdio.h>

//constant
const int N = 3;
//must copy first line of function up top to declare it to C
//known as a prototype
float average(int length, int array[]);


int main(void)
{
    int scores[N];
    for (int i = 0; i < N; i++)
    {
        scores[i] = get_int("Score: ");
    }

    //printf("Average: %f\n", (scores[0] + scores[1] + scores[2]) / (float) 3);
    printf("Average: %f\n", average(N, scores));
}
//be sure to include 'int array[]' as function placeholder input to specify argument as an array
//arrays must be composed of integers
float average(int length, int array[])
{
    int sum = 0;
    for (int i = 0; i < length; i++)
    {
        sum += array[i];
    }
    return sum / (float) length;
}