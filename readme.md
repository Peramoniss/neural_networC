# Neural NetworC
Neural NetworC is a MLP (Multilayer Perceptron) library implementation for C programming language. Neural NetworC contains two main modules: the MLP module, encharged of neural network operations, and the Bamboo module, which handles some data transformation for the network. New modules with different neural network architectures may arrive as my madness continues to deepen. Currently, the library is in alpha version, so bugs may arise and some features might not be generic enough. Open issues with suggestions or pull a request to collaborate.

## Current Limitations:
1. The dataset must contain only numeric data (integer and float/double). It will handle string data as 0, and therefore not cause errors, but will not behave properly.
2. The dataset must be on dummy format for the get_result_report function to work. If you want to use a different encoding, you must implement the results handling by yourself. Later versions might be more generic or offer more variation.

## Installing
To compile the library, run the following command on terminal (make sure you're in the same folder as the Makefile archive):

```bash
make
```

This command generates files `bamboo.o`, `MLP.o`, and `libneuralnetworc.a`. After that, to compile your program that uses the library, run the following command, knowing that:
* `-I` indicates the path to the .h files
* `examples/main.c` is the path to the file that's being compiled
* `-L` indicates the path to the library file
* `-lneuralnetworc` indicates that the library being used is defined in the file libneuralnetwork.a (which was generated when you compiled the library)
* `-o` indicates that the next parameter (your_program) will be the name of the executable file generated

```bash
gcc -I./include examples/main.c -L. -lneuralnetworc -o your_program
```

To run it after, you can just use the following command:

```bash
./your_program
```

Alternatively, you can use a bash script to make the process easier for you. The file **nnc** (in linux) and **nnc.ps1** (in powershell terminal) will allow you to compile your program by running the following command, with as many gcc options as you want:

```bash
./nnc examples/main.c -o program
```

## Library Documentation
### Bamboo Module
#### Structs
```c
typedef struct dataset
{
    double **matrix; 
    int columns;     
    int rows;
} dataset;
```
#### Enums
```c
enum split_method{
    PEREIRA,
    FISCHER_YATES
};
```

#### Functions


```
dataset *load_csv(FILE *file, char *delim, short header, int row_max_size, short force_not_null)
```
* Reads a csv file defined in `file`, considering the delimiter char in `delim`, and returns a dataset. `header` defines wheter there's a header (1) or not (0), `row_max_size` defines the maximum amount of characters a line might have, and `force_not_null` decides if the dataset can consider NULL data as 0 or stop the reading if it encounters a NULL data.

```
void min_max_scale_column(dataset *data, int pos)
```
* Apply Min-Max Scaling in the column of index equals to `pos` in the dataset provided in `data`.

```
void min_max_scale_data(dataset *data)
```
* Apply Min-Max Scaling in in the dataset provided in `data`.


```
void x_y_split(dataset *original, dataset **x, dataset **y, int output_size);
```
* Divides the dataset pointed by `original` in x and y, and returns the splitted dataset in two other datasets `x` and `y`, sent as parameters. `x` will contain columns from index 0 to the number of columns in original minus `output_size` minus 1, and y will contain the last `output_size` columns in original.

```
void train_test_split(dataset *data, dataset **train, dataset **test, double train_ratio, int random_state, int output_size, enum split_method method);
```

* Divides the dataset in `data` into training and testing datasets, returned in `train` and `test` variables. `train_ratio` must be a number between 0 and 1 (not included), and it defines the percentage of instances that will be alocated to training and testing. 0.7 to 0.8 are recommended values. `random_state` defines the random seed in the instance shuffling. `output_size` defines where the classes are encountered, as the method is stratified (tries to maintain the proportion between the classes). `method` defines what shuffling method will be used. `PEREIRA` method is slower but __tends__ to get better results, while `FISCHER_YATES` is very simple but fast.

```
void to_csv(dataset *data, char *dir);
```
* Creates a csv file of the dataset in `data` and saves it in the directory defined in `dir`. 

```
void dataframe_head(dataset *data, int head);
```
* Writes in console the `head` first lines in the dataframe in `data`.

### MLP Module
#### Structs

```
typedef struct weight_matrix
{
    double **matrix;
    int columns;    
    int rows;
} weight_matrix;
```

```
typedef struct metrics
{
    double global_accuracy;
    int classes;
    int **confusion_matrix;
    double **class_metrics; //rows equivalent to classes, 3 columns - precision, recall, f-1. 
} metrics;
```

```
typedef struct layer
{
    double *values; 
    int neurons; 
} layer;          
```

```
typedef struct neural_network
{
    layer *layers;
    layer *errors;
    weight_matrix *weights;
    metrics *report;
    int input_neurons;
    int output_neurons;
    int hidden_layers;
    int total_layers;
    enum activation_function activation;
} neural_network;
```

#### Enums
```
enum activation_function {
    SIGMOID,
    RELU,
    TANH
};
```

#### Functions
```
void get_dummies(dataset *data);
```
* Identify every categorical column in `data` and uses dummy codification in it. 

```
int dummy(dataset *data, int column);
```
* Uses dummy codification in the column of index `column` in dataset `data`.

```
neural_network *create_neural_network(dataset *data, int output_neurons, int hidden_layers, int *hidden_layers_size, enum activation_function activation_function, int random_state);
```
* Creates and returns a neural network using the struct `neural_network`. The input layer will be based on the dataset you send as parameter in `data`. The outpur layer will be defined by the `output_neurons` parameter. `hidden_layers` is an integer defining how many hidden layers there will be, while `hidden_layers_size` is an integer array containing in each position how many neurons each of the hidden layers will have. If NULL, it will calculate automatically the size (not guaranteed to make good predictions). `activation_function` defines which of the enumerated options will be used as activation function in the trainment (`SIGMOID` recommended). `random_state` defines the random seed for matrix initialization.

```
void train(neural_network *network, dataset *x_data, dataset *y_data, double learning_rate, double momentum, int epochs, int monitor);
```
* Trains the neural network in `network` with the data in `x_data`, comparing the results obtained in each instance with the ones in `y_data` (expected target values). `learning_rate` receives an integer between 0 and 1 that defines how fast the network learns (0.1 recommended). `momentum` defines a multiplier in the network, and is also a value between 0 and 1. `epochs` defines for how many epochs the data will be trained, while `monitor` defines how often the MSE loss will be shown in the screen (monitor in monitor epochs). The macro `fit` expands to this function.

```
void train_with_early_stopping(neural_network *network, dataset *x_data, dataset *y_data, double learning_rate, double momentum, int epochs, int monitor, double min_loss, int patience);
```
* Works just like `train`, but stops early if the algorhythm sees there's not enough gain in the MSE loss metric. `min_loss` defines what is the minimum difference between two monitored losses. If the difference is lower than that for `patience` monitored epochs, the algorhythm supposes there's not enough progress being made and stops training. The macro `fit_with_early_stopping` expands to this function.

```
void get_results_report(neural_network *network, dataset *x, dataset *y, short log);
```
* Generates a report of how well the neural network in `network` predicts the data in `x`, considering the correct values in `y`. The results will be attributed to the `metrics` field in `network`. If `log` is different than 0, the function will also print the results. 

```
void print_confusion_matrix(metrics *report);
```
* Prints the confusion matrix after in `report`. 

```
int *predict(double **x, double **y, int instances, neural_network *network, int log);
```
* Using the neural network in `network`, predicts every instance of `x` and returns an array, with every position being what it predicted to the instance in the index. `x` must be a matrix with columns equivalent to the number of input neurons in the network. If `log` is different than 0, the function will print the predicted class and the expected class, available in `y` (with the same amount of rows as x, and columns equivalent to the number of output neurons). If `log` is 0, `y` is irrelevant and can be NULL.

```
void export_neural_network(neural_network *nn, char *dir);
```
* Creates a txt file with the settings for the neural network in `nn` and saves it in the directory defined in `dir`. 

```
neural_network * load_neural_network(FILE *source);
```
* Loads the neural network in the txt file pointed by `source`.

## Usage Example

```
#include <stdio.h>
#include <stdlib.h>
#include "../include/bamboo.h"
#include "../include/MLP.h"

int main(){
    FILE *file = fopen("examples/data.csv", "r"); //reads the csv file 
    dataset *data = load_csv(file, ",", 1, 1000, 1); //transfer it to the dataset structure
    min_max_scale_data(data); //scale the dataset
    get_dummies(data); //get the dummy codification for the data

    int output_neurons = 2, hidden_layers = 2; //defines the structure of the newtwork
    dataset *test_data_splitted, *train_data_splitted; //creates datasets for the train and test datasets
    train_test_split(data, &train_data_splitted, &test_data_splitted, 0.7, 39, output_neurons, PEREIRA); //divide the dataset in train and test

    int layers_size[] = {20,10};
    dataset *x, *y;
    x_y_split(train_data_splitted, &x, &y, output_neurons); //splits train_data_splitted in x and y
    neural_network *nn = create_neural_network(train_data_splitted, output_neurons, hidden_layers, layers_size, SIGMOID, NULL); //creates the neural network with random (NULL) seed

    fit_with_early_stopping(nn, x, y, 0.1, 1, 1000, 10, 0.0001, 1); //train with early stoppping

    dataset *test_x, *test_y; 
    x_y_split(test_data_splitted, &test_x, &test_y, output_neurons); //splits test_data_splitted in x and y

    get_results_report(nn, test_x, test_y, 1); //tests the network and uses the log to see the results 
    print_confusion_matrix(nn->report); //prints the confusion matrix as well

    export_neural_network(nn, "results/exported.txt"); //generates a txt file with the network so it can be loaded later
    
    return 0;
}
```