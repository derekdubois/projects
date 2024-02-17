import csv
import sys


def main():

    # TODO: Check for command-line usage
    if len(sys.argv) != 3:
        sys.exit("Please provide valid CSV and text files")

    # TODO: Read database file into a variable
    csv_list = []
    csv_file = sys.argv[1]
    with open(csv_file, 'r') as my_file:
        dict_file = csv.DictReader(my_file)
        for name in dict_file:
            csv_list.append(name)



    # TODO: Read DNA sequence file into a variable


    dna_file = sys.argv[2]
    with open(dna_file, 'r') as my_dna_file:
        dna_sequence = my_dna_file.read()

    # TODO: Find longest match of each STR in DNA sequence
    subsequences = list(csv_list[0].keys())[1:]

    output = {}

    for subsequence in subsequences:
        output[subsequence] = longest_match(dna_sequence, subsequence)


    # TODO: Check database for matching profiles
    for entry in csv_list:
        match = 0
        for subsequence in subsequences:
            if int(entry[subsequence]) == output[subsequence]:
                match += 1

        if match == len(subsequences):
            print(entry["name"])
            return
    print("No match")

def longest_match(sequence, subsequence):
    """Returns length of longest run of subsequence in sequence."""

    # Initialize variables
    longest_run = 0
    subsequence_length = len(subsequence)
    sequence_length = len(sequence)

    # Check each character in sequence for most consecutive runs of subsequence
    for i in range(sequence_length):

        # Initialize count of consecutive runs
        count = 0

        # Check for a subsequence match in a "substring" (a subset of characters) within sequence
        # If a match, move substring to next potential match in sequence
        # Continue moving substring and checking for matches until out of consecutive matches
        while True:

            # Adjust substring start and end
            start = i + count * subsequence_length
            end = start + subsequence_length

            # If there is a match in the substring
            if sequence[start:end] == subsequence:
                count += 1

            # If there is no match in the substring
            else:
                break

        # Update most consecutive matches found
        longest_run = max(longest_run, count)

    # After checking for runs at each character in seqeuence, return longest run found
    return longest_run


main()
