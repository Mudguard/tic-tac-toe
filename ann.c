#include "ann.h"

// double activation_function(double x)
// {
//     return 1.0 / (1.0 + exp(-x));
// }

// double activation_derivative(double a)
// {
//     return a * (1.0 - a);
// }

double activation_function(double x)
{
    if (x > 0)
        return x;
    else
        return 0;
}

double activation_derivative(double a)
{
    if (a < 0)
        return 0;
    else
        return 1;
}

gsl_matrix *initialize_matrix_rand(int num_in, int num_out)
{
    gsl_matrix *tmp = gsl_matrix_alloc(num_in, num_out);
    for (int i = 0; i < num_in; i++)
    {
        for (int j = 0; j < num_out; j++)
        {
            gsl_matrix_set(tmp, i, j, (double)rand() / RAND_MAX * 1.0 - 0.5);
        }
    }
    return tmp;
}

void add_layer(struct layer *ann, int layer_num, int batch_size, int num_in, int num_out)
{
    (ann + layer_num)->num_in = num_in;
    (ann + layer_num)->num_out = num_out;
    (ann + layer_num)->weight = initialize_matrix_rand(num_in, num_out);
    (ann + layer_num)->bias = initialize_matrix_rand(1, num_out);
    (ann + layer_num)->sigma = gsl_matrix_calloc(batch_size, num_out);
    (ann + layer_num)->activation = gsl_matrix_calloc(batch_size, num_out);
    (ann + layer_num)->dEda = gsl_matrix_calloc(batch_size, num_out);
    (ann + layer_num)->dads = gsl_matrix_calloc(batch_size, num_out);
    (ann + layer_num)->gradw = gsl_matrix_calloc(num_in, num_out);
    (ann + layer_num)->gradb = gsl_matrix_calloc(1, num_out);
}

void compute_activation(struct layer *n)
{
    for (int i = 0; i < n->sigma->size1; i++)
    {
        for (int j = 0; j < n->sigma->size2; j++)
        {
            gsl_matrix_set(n->sigma, i, j, gsl_matrix_get(n->sigma, i, j) + gsl_matrix_get(n->bias, 0, j));
            gsl_matrix_set(n->activation, i, j, activation_function(gsl_matrix_get(n->sigma, i, j)));
        }
    }
}

void compute_dads(struct layer *n)
{
    for (int i = 0; i < n->sigma->size1; i++)
    {
        for (int j = 0; j < n->sigma->size2; j++)
        {
            gsl_matrix_set(n->dads, i, j, activation_derivative(gsl_matrix_get(n->activation, i, j)));
        }
    }
}

void compute_gradb(struct layer *n)
{
    for (int j = 0; j < n->dads->size2; j++)
    {
        double sum = 0;
        for (int i = 0; i < n->dads->size1; i++)
        {
            sum += gsl_matrix_get(n->dads, i, j);
        }
        gsl_matrix_set(n->gradb, 0, j, sum / (double)n->dads->size1);
    }
}

void initialize_input_layer(struct layer *ann, gsl_matrix *input)
{
    gsl_matrix_memcpy(ann->activation, input);
}

void forward_propagation(struct layer *ann, int num_layers, int batch_size, gsl_matrix *input)
{
    for (int i = 1; i < num_layers; i++)
    {
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, (ann + i - 1)->activation, (ann + i)->weight, 0.0, (ann + i)->sigma);
        compute_activation(ann + i);
    }
}  

void backward_propagation(struct layer *ann, int num_layers, int batch_size, gsl_matrix *target)
{
    double eta = 0.001;
    gsl_matrix_memcpy((ann + num_layers - 1)->dEda, (ann + num_layers - 1)->activation);
    gsl_matrix_sub((ann + num_layers - 1)->dEda, target);

    for (int i = num_layers - 1; i > 0; i--)
    {
        compute_dads(ann + i);
        gsl_matrix_mul_elements((ann + i)->dads, (ann + i)->dEda);
        gsl_blas_dgemm (CblasTrans , CblasNoTrans, 1.0, (ann + i - 1)->activation, (ann + i)->dads, 0.0, (ann + i)->gradw);
        compute_gradb(ann + i);
        gsl_matrix_scale((ann + i)->gradw, -eta);
        gsl_matrix_scale((ann + i)->gradb, -eta);
        gsl_matrix_add((ann + i)->weight, (ann + i)->gradw);
        gsl_matrix_add((ann + i)->bias, (ann + i)->gradb);
        gsl_blas_dgemm (CblasNoTrans , CblasTrans, 1.0, (ann + i)->dads, (ann + i)->weight, 0.0, (ann + i - 1)->dEda);
    }
}

void print_output(struct layer *ann, int num_layers, int batch_size)
{
    for (int i = 0; i < batch_size; i++)
    {
        for (int j = 0; j < (ann + num_layers - 1)->num_out; j++)
        {
            printf("%f ", gsl_matrix_get((ann + num_layers - 1)->activation, i, j));
        }
        printf("\n");
    }
}

void free_ann(struct layer *ann, int num_layers)
{
    for (int i = 0; i < num_layers; i++)
    {
        free((ann + i)->weight);
        free((ann + i)->bias);
        free((ann + i)->sigma);
        free((ann + i)->activation);
        free((ann + i)->dEda);
        free((ann + i)->dads);
        free((ann + i)->gradw);
        free((ann + i)->gradb);
    }
    free(ann);
}
