#ifndef LQR_H
#define LQR_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

// LQR Controller state
typedef struct {
    gsl_matrix *A; // System dynamics matrix
    gsl_matrix *B; // Control input matrix
    gsl_matrix *Q; // Cost for state deviation
    gsl_matrix *R; // Cost for control effort
    gsl_matrix *P; // Solution to Riccati equation (cost-to-go)
    gsl_matrix *K; // Optimal feedback gain matrix
    int state_size; // Size of the state vector
    int control_size; // Size of the control input vector
} LQR;

// Function prototypes
void lqr_init(LQR *lqr, int state_size, int control_size);
void lqr_free(LQR *lqr);
void lqr_compute_gain(LQR *lqr);
void lqr_set_system_dynamics_A(LQR *lqr, const gsl_matrix *A);
void lqr_set_system_dynamics_B(LQR *lqr, const gsl_matrix *B);
void lqr_set_cost_matrix_Q(LQR *lqr, const gsl_matrix *Q);
void lqr_set_cost_matrix_R(LQR *lqr, const gsl_matrix *R);
void lqr_compute_control_input(LQR *lqr, const gsl_vector *x, gsl_vector *u);
void lqr_process(LQR *lqr, const gsl_vector *x, gsl_vector *u);

#endif // LQR_H
