#include <stdio.h>

int main(void)
{
    int n = 50;
    //to declare the pointer, use the * before the variable; the & before n declares is as where n is in memory
    int *p = &n;
    // to go to whatever address p is storing, use * before p in the printf function
    printf("%i\n", *p);
}