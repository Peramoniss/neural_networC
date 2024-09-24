// Function to delete a column from the original matrix and substitute it with new columns
int** substitute_columns(int** original_matrix, int rows, int orig_cols, int** new_matrix, int new_cols, int col_position) {
    // Allocate memory for the new matrix with adjusted column size
    int new_total_cols = orig_cols - 1 + new_cols;
    int** new_result = (int**)malloc(rows * sizeof(int*));
    
    for (int i = 0; i < rows; i++) {
        new_result[i] = (int*)malloc(new_total_cols * sizeof(int));
    }

    // Copy columns before the insertion point
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < col_position; j++) {
            new_result[i][j] = original_matrix[i][j];
        }
    }

    // Insert new columns from the new matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            new_result[i][col_position + j] = new_matrix[i][j];
        }
    }

    // Copy remaining columns from the original matrix, skipping the deleted column
    for (int i = 0; i < rows; i++) {
        for (int j = col_position + 1; j < orig_cols; j++) {
            new_result[i][j - 1 + new_cols] = original_matrix[i][j];
        }
    }

    return new_result;
}