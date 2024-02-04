#include "KalmanFilter.h"
#include <gsl/gsl_blas.h>

// Initialize the Kalman Filter
void kalman_init(KalmanFilter *kf, int state_size, int control_size, int measurement_size) {
    // Allocate matrices and vectors
    kf->A = gsl_matrix_alloc(state_size, state_size);
    kf->B = gsl_matrix_alloc(state_size, control_size);
    kf->H = gsl_matrix_alloc(measurement_size, state_size);
    kf->Q = gsl_matrix_alloc(state_size, state_size);
    kf->R = gsl_matrix_alloc(measurement_size, measurement_size);
    kf->x = gsl_vector_alloc(state_size);
    kf->P = gsl_matrix_alloc(state_size, state_size);
    // Set system dynamics for a spring-mass system
    double dt = 1.0; // Sample time
    double k = 1.0; // Spring constant
    double m = 1.0; // Mass
    gsl_matrix_set(kf->A, 0, 0, 1 - k/m * dt);
    gsl_matrix_set(kf->B, 0, 0, dt / m);
    gsl_matrix_set_identity(kf->H);
    gsl_matrix_set(kf->Q, 0, 0, 0.01); // Process noise covariance
    gsl_matrix_set(kf->R, 0, 0, 0.1); // Measurement noise covariance
    // Set initial state and covariance estimates
    gsl_vector_set_zero(kf->x);
    gsl_matrix_set_identity(kf->P);
}


// Free the Kalman Filter resources
void kalman_free(KalmanFilter *kf) {
    gsl_matrix_free(kf->A);
    gsl_matrix_free(kf->B);
    gsl_matrix_free(kf->H);
    gsl_matrix_free(kf->Q);
    gsl_matrix_free(kf->R);
    gsl_matrix_free(kf->P);
    gsl_matrix_free(kf->K);
    gsl_vector_free(kf->x);
}

// Predict the state and estimate covariance
void kalman_predict(KalmanFilter *kf, const gsl_vector *u) {
    // x = A*x + B*u
    gsl_vector *tmp_x = gsl_vector_alloc(kf->x->size);
    gsl_blas_dgemv(CblasNoTrans, 1.0, kf->A, kf->x, 0.0, tmp_x); // A*x
    gsl_blas_dgemv(CblasNoTrans, 1.0, kf->B, u, 1.0, tmp_x); // + B*u
    gsl_vector_memcpy(kf->x, tmp_x);
    gsl_vector_free(tmp_x);

    // P = A*P*A' + Q
    gsl_matrix *tmp_P = gsl_matrix_alloc(kf->P->size1, kf->P->size2);
    gsl_matrix_memcpy(tmp_P, kf->P);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, kf->A, tmp_P, 0.0, kf->P); // A*P
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, kf->P, kf->A, 0.0, tmp_P); // *A'
    gsl_matrix_add(tmp_P, kf->Q); // + Q
    gsl_matrix_memcpy(kf->P, tmp_P);
    gsl_matrix_free(tmp_P);
}

// Update the state estimate with a new measurement
void kalman_update(KalmanFilter *kf, const gsl_vector *z) {
    // K = P*H'*inv(H*P*H' + R)
    gsl_matrix *tmp_PHt = gsl_matrix_alloc(kf->P->size1, kf->H->size1);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, kf->P, kf->H, 0.0, tmp_PHt); // P*H'
    gsl_matrix *tmp_HPHtR = gsl_matrix_alloc(kf->H->size1, kf->H->size1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, kf->H, tmp_PHt, 0.0, tmp_HPHtR); // H*P*H'
    gsl_matrix_add(tmp_HPHtR, kf->R); // + R
    gsl_matrix *inv_HPHtR = gsl_matrix_alloc(tmp_HPHtR->size1, tmp_HPHtR->size2);
    gsl_permutation *p = gsl_permutation_alloc(tmp_HPHtR->size1);
    int signum;
    gsl_linalg_LU_decomp(tmp_HPHtR, p, &signum);
    gsl_linalg_LU_inv(tmp_HPHtR, p, inv_HPHtR);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp_PHt, inv_HPHtR, 0.0, kf->K); // K = P*H'*inv(H*P*H' + R)
    gsl_matrix_free(tmp_PHt);
    gsl_matrix_free(tmp_HPHtR);
    gsl_matrix_free(inv_HPHtR);
    gsl_permutation_free(p);

    // x = x + K*(z - H*x)
    gsl_vector *tmp_zHx = gsl_vector_alloc(z->size);
    gsl_blas_dgemv(CblasNoTrans, -1.0, kf->H, kf->x, 0.0, tmp_zHx); // -H*x
    gsl_vector_add(tmp_zHx, z); // z - H*x
    gsl_blas_dgemv(CblasNoTrans, 1.0, kf->K, tmp_zHx, 1.0, kf->x); // x + K*(z - H*x)
    gsl_vector_free(tmp_zHx);

    // P = (I - K*H)*P
    gsl_matrix *tmp_KH = gsl_matrix_alloc(kf->K->size1, kf->H->size2);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, kf->K, kf->H, 0.0, tmp_KH); // K*H
    gsl_matrix *I = gsl_matrix_alloc(kf->P->size1, kf->P->size2);
    gsl_matrix_set_identity(I);
    gsl_matrix_sub(I, tmp_KH); // I - K*H
    gsl_matrix *tmp_IP = gsl_matrix_alloc(I->size1, kf->P->size2);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, I, kf->P, 0.0, tmp_IP); // (I - K*H)*P
    gsl_matrix_memcpy(kf->P, tmp_IP);
    gsl_matrix_free(tmp_KH);
    gsl_matrix_free(I);
    gsl_matrix_free(tmp_IP);
}

// Function to process a measurement through the Kalman Filter
void kalman_process(KalmanFilter *kf, const gsl_vector *u, const gsl_vector *z) {
    // Predict the next state
    kalman_predict(kf, u);

    // Update the state with the new measurement
    kalman_update(kf, z);
}

void kalman_set_system_dynamics_A(KalmanFilter *kf, const gsl_matrix *A) {
    gsl_matrix_memcpy(kf->A, A);
}

void kalman_set_system_dynamics_B(KalmanFilter *kf, const gsl_matrix *B) {
    gsl_matrix_memcpy(kf->B, B);
}

void kalman_set_observation_model_H(KalmanFilter *kf, const gsl_matrix *H) {
    gsl_matrix_memcpy(kf->H, H);
}

void kalman_set_process_noise_covariance_Q(KalmanFilter *kf, const gsl_matrix *Q) {
    gsl_matrix_memcpy(kf->Q, Q);
}

void kalman_set_measurement_noise_covariance_R(KalmanFilter *kf, const gsl_matrix *R) {
    gsl_matrix_memcpy(kf->R, R);
}

void kalman_set_initial_state_x0(KalmanFilter *kf, const gsl_vector *x0) {
    gsl_vector_memcpy(kf->x, x0);
}

void kalman_set_initial_estimate_error_covariance_P0(KalmanFilter *kf, const gsl_matrix *P0) {
    gsl_matrix_memcpy(kf->P, P0);
}

