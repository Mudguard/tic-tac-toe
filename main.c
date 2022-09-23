#include <time.h>

#include "ann.h"

int main()
{
    srand(time(NULL));
    int batch_size = 8;
    int num_layers = 3;
    struct layer *ann = calloc(num_layers, sizeof(struct layer));
    add_layer(ann, 0, batch_size, 0, 3);
    add_layer(ann, 1, batch_size, 3, 7);
    add_layer(ann, 2, batch_size, 7, 3);

    double values[] = 
    {
        0, 0, 0,
        0, 0, 1,
        0, 1, 0,
        0, 1, 1,
        1, 0, 0,
        1, 0, 1,
        1, 1, 0,
        1, 1, 1
    };

    gsl_matrix_view input = gsl_matrix_view_array(values, 8, 3);
    gsl_matrix_view target = gsl_matrix_view_array(values, 8, 3);
    
    printf("\e[?25l"); //hide cursor.
    printf("\x1b[H\x1b[J"); //clear screen.

    size_t iterations = 300000;
    initialize_input_layer(ann, &input.matrix);
    for (size_t i = 0; i < iterations; i++)
    {
        forward_propagation(ann, num_layers, batch_size, &input.matrix);
        backward_propagation(ann, num_layers, batch_size, &target.matrix);
        printf("\033[%d;%dH", 1, 1); //cursor position
        print_output(ann, num_layers, batch_size);
    }

    free_ann(ann, num_layers);
    return 0;
}
