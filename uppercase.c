#include <cs50.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
//converts string to uppercase
//C treats the string data as ASCII values, allowing them to be converted to other characters using integers
//int main(void)
//{
  //  string s = get_string("Before: ");
    //printf("After: ");
    //for (int i = 0; i < strlen(s); i++)
    //{
        //if (s[i] >= 'a' && s[i] <= 'z')
        //if (islower(s[i]))
        //{
            //printf("%c", toupper(s[i]));
            //printf("%c", s[i] - 32);
      //  }
        //else
        //{
        //    printf("%c", s[i]);
        //}
   // }
   // printf("\n");
//}

int main(void)
{
    string s = get_string("Before: ");
    printf("After: ");
    //notice how you can declare more than one variable at beginning of for loop
    for (int i = 0, n = strlen(s); i < n; i++)
    {
        printf("%c", toupper(s[i]));
    }
    printf("\n");
}