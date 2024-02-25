#include <cs50.h>
#include <stdio.h>

int main(void)
{
    int n;
    // Prompt for start size
    do
    {
        n = get_int("Starting population size: ");
    }
    while (n < 9);

    // Prompt for end size
    int p;
    do
    {
        p = get_int("Ending population size: ");
    }
    while (p < n);

    // Calculate number of years until we reach threshold
    int y;
    for (y = 0; n < p; y++)
    {
        n = n + (n / 3) - (n / 4);
    }



    // Print number of years

    printf("Years: %i\n", y);
}
