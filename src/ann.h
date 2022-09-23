#ifndef __ANN_H__
#define __ANN_H__

#include <stdio.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_blas.h>

struct layer
{
    int num_in;
    int num_out;
    gsl_matrix *weight;
    gsl_matrix *bias;
    gsl_matrix *sigma;
    gsl_matrix *activation;
    gsl_matrix *dEda;
    gsl_matrix *dads;
    gsl_matrix *gradw;
    gsl_matrix *gradb;
};

void add_layer(struct layer *ann, int layer_num, int batch_size, int num_in, int num_out);
void initialize_input_layer(struct layer *ann, gsl_matrix *input);
void forward_propagation(struct layer *ann, int num_layers, int batch_size, gsl_matrix *input);
void backward_propagation(struct layer *ann, int num_layers, int batch_size, gsl_matrix *target);
void print_output(struct layer *ann, int num_layers, int batch_size);
void free_ann(struct layer *ann, int num_layers);

#endif //__ANN_H__
