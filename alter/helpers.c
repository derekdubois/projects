#include <math.h>
#include "helpers.h"

#define COLOR_RED 0
#define COLOR_GREEN 1
#define COLOR_BLUE 2

// Convert image to grayscale
void grayscale(int height, int width, RGBTRIPLE image[height][width])
{
    for (int row = 0; row < height; row++)
    {
        for (int column = 0; column < width; column++)
        {
            int g_scale = round((image[row][column].rgbtBlue + image[row][column].rgbtGreen + image[row][column].rgbtRed) / 3.0);

            image[row][column].rgbtBlue = g_scale;
            image[row][column].rgbtGreen = g_scale;
            image[row][column].rgbtRed = g_scale;
        }
    }
    return;
}

// Convert image to sepia
void sepia(int height, int width, RGBTRIPLE image[height][width])
{
    for (int row = 0; row < height; row++)
    {
        for (int column = 0; column < width; column++)
        {
            int sepiaRed = round(.393 * image[row][column].rgbtRed + .769 * image[row][column].rgbtGreen + .189 * image[row][column].rgbtBlue);
            int sepiaGreen = round(.349 * image[row][column].rgbtRed + .686 * image[row][column].rgbtGreen + .168 * image[row][column].rgbtBlue);
            int sepiaBlue = round(.272 * image[row][column].rgbtRed + .534 * image[row][column].rgbtGreen + .131 * image[row][column].rgbtBlue);

            image[row][column].rgbtRed = fmin(255, sepiaRed);
            image[row][column].rgbtGreen = fmin(255, sepiaGreen);
            image[row][column].rgbtBlue = fmin(255, sepiaBlue);
        }
    }
    return;
}

// Reflect image horizontally
void reflect(int height, int width, RGBTRIPLE image[height][width])
{
    RGBTRIPLE placeholder;

    for (int row = 0; row < height; row++)
    {
        for (int column = 0; column < width / 2; column++)
        {
            placeholder = image[row][column];
            image[row][column] = image[row][width - column - 1];
            image[row][width - column - 1] = placeholder;
        }
    }
    return;
}

int Blurfunc(int i, int j, int height, int width, RGBTRIPLE image[height][width], int color_position)
{
    float pixel_count = 0;
    int total = 0;
    for (int row = i -1; row <= (i + 1); row++)
    {
        for (int column = j - 1; column <= (j + 1); column++)
        {
            if (row < 0 || row >= height || column < 0 || column >= width)
            {
                continue;
            }
            if (color_position == COLOR_RED)
            {
                total += image[row][column].rgbtRed;
            }
            else if (color_position == COLOR_GREEN)
            {
                total += image[row][column].rgbtGreen;
            }
            else
            {
                total += image[row][column].rgbtBlue;
            }
            pixel_count++;
        }
    }
    return round(total / pixel_count);
}
// Blur image
void blur(int height, int width, RGBTRIPLE image[height][width])
{
    RGBTRIPLE duplicate[height][width];
    for (int row = 0; row < height; row++)
    {
        for (int column = 0; column < width; column++)
        {
            duplicate[row][column] = image[row][column];
        }
    }
    for (int row = 0; row < height; row++)
    {
        for (int column = 0; column < width; column++)
        {
            image[row][column].rgbtRed = Blurfunc(row, column, height,  width, duplicate, COLOR_RED);
            image[row][column].rgbtGreen = Blurfunc(row, column, height,  width, duplicate, COLOR_GREEN);
            image[row][column].rgbtBlue = Blurfunc(row, column, height,  width, duplicate, COLOR_BLUE);
        }
    }

    return;
}
