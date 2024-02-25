#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "wav.h"

int check_format(WAVHEADER header);
int get_block_size(WAVHEADER header);


int main(int argc, char *argv[])
{
    // Ensure proper usage
    if (argc != 3)
    {
        printf("Usage: ./reverse input.wav output.wav\n");
        return 1;
    }

    // Open input file for reading
    FILE *input = fopen(argv[1], "r");
    if (input == NULL)
    {
        printf("Could not open file.\n");
        return 1;
    }
    // Read header
    const int HEADER_SIZE = sizeof(WAVHEADER);
    WAVHEADER header;
    fread(&header, HEADER_SIZE, 1, input);

    // Use check_format to ensure WAV format
    if (check_format(header) == 1)
    {
        return 1;
    }

    // Open output file for writing
    FILE *output = fopen(argv[2], "w");
    if (output == NULL)
    {
        printf("Could not open file.\n");
        return 1;
    }
    // Write header to file
    fwrite(&header, HEADER_SIZE, 1, output);

    // Use get_block_size to calculate size of block
    int block_size = get_block_size(header);

    // Write reversed audio to file
    if (fseek(input, block_size, SEEK_END))
    {
        return 1;
    }
    BYTE audio_buffer[block_size];
    while (ftell(input) - block_size > sizeof(header))
    {
        if (fseek(input, -2 * block_size, SEEK_CUR))
        {
            return 1;
        }
        fread(audio_buffer, block_size, 1, input);
        fwrite(audio_buffer, block_size, 1, output);
    }
    fclose(input);
    fclose(output);
}

int check_format(WAVHEADER header)
{
    if (header.format[0] == 'W' && header.format[1] == 'A' && header.format[2] == 'V' && header.format[3] == 'E')
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

int get_block_size(WAVHEADER header)
{
    int block_size = 0;
    block_size = header.numChannels * (header.bitsPerSample * 0.125);
    return block_size;
}