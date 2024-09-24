#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define a struct for the one-hot encoded data
typedef struct {
    int rows;
    int cols;
    int** data; // A pointer to hold the one-hot encoded matrix
} OneHotEncoded;

// Function to find the unique classes in the classification column
int find_unique_classes(char** labels, int num_samples, char** unique_classes) {
    int unique_count = 0;

    // Loop through all samples
    for (int i = 0; i < num_samples; i++) {
        int is_unique = 1;

        // Check if the label is already in unique_classes
        for (int j = 0; j < unique_count; j++) {
            if (strcmp(labels[i], unique_classes[j]) == 0) {
                is_unique = 0;
                break;
            }
        }

        // If it is a new unique class, add it
        if (is_unique) {
            unique_classes[unique_count] = labels[i];
            unique_count++;
        }
    }
    return unique_count;
}

// Function to perform one-hot encoding
OneHotEncoded one_hot_encode(char** labels, int num_samples, char** unique_classes, int num_classes) {
    // Create the one-hot encoding matrix
    OneHotEncoded encoding;
    encoding.rows = num_samples;
    encoding.cols = num_classes;

    // Allocate memory for the one-hot matrix
    encoding.data = (int**)malloc(num_samples * sizeof(int*));
    for (int i = 0; i < num_samples; i++) {
        encoding.data[i] = (int*)calloc(num_classes, sizeof(int));
    }

    // Assign one-hot encoding based on class index
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_classes; j++) {
            if (strcmp(labels[i], unique_classes[j]) == 0) {
                encoding.data[i][j] = 1; // Set the corresponding class index to 1
            }
        }
    }

    return encoding;
}

// Function to print the one-hot encoded matrix
void print_one_hot_encoded(OneHotEncoded encoding) {
    for (int i = 0; i < encoding.rows; i++) {
        for (int j = 0; j < encoding.cols; j++) {
            printf("%d ", encoding.data[i][j]);
        }
        printf("\n");
    }
}

// Main program
int main() {
    // Example classification column with labels
    char* labels[] = {"cat", "dog", "dog", "fish", "cat", "bird"};
    int num_samples = 6;

    // Create an array to hold the unique classes (assuming no more than num_samples unique classes)
    char* unique_classes[num_samples];
    
    // Find the unique classes
    int num_classes = find_unique_classes(labels, num_samples, unique_classes);

    // Perform one-hot encoding
    OneHotEncoded encoding = one_hot_encode(labels, num_samples, unique_classes, num_classes);

    // Print the one-hot encoded matrix
    printf("One-hot encoded matrix:\n");
    print_one_hot_encoded(encoding);

    // Free memory
    for (int i = 0; i < num_samples; i++) {
        free(encoding.data[i]);
    }
    free(encoding.data);

    return 0;
}
