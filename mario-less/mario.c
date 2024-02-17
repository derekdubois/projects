#include <cs50.h>
#include <stdio.h>

int main(void)
{
    int n;
    //do while loop
    do
    {
        n = get_int("Height: ");
    }
    while (n < 1 || n > 8);
    //to make the block grid
    for (int i = 0; i < n; i++)
    {
        //to add spaces
        for (int j = n - i; j > 1; j--)
        {
            printf(" ");
        }
        //to add hashes
        for (int m = 0; m <= i; m++)
        {
            printf("#");
        }
        printf("\n");
    }
}