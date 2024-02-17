#include <stdio.h>
#include <cs50.h>

int main(void)
{
    int n;
    //do while loop
    do
    {
        n = get_int("Size: ");
    }
    while (n < 1);
    //to make the block grid
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("#");
        }
        printf("\n");
    }
}