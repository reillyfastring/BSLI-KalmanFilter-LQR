#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

// Kalman Filter state
typedef struct {
    gsl_matrix *A; // State transition model
    gsl_matrix *B; // Control input model
    gsl_matrix *H; // Observation model
    gsl_matrix *Q; // Process noise covariance
    gsl_matrix *R; // Measurement noise covariance
    gsl_matrix *P; // Estimate error covariance
    gsl_vector *x; // State estimate
    gsl_matrix *K; // Kalman gain
} KalmanFilter;

// Function prototypes
void kalman_init(KalmanFilter *kf, int state_size, int control_size, int measurement_size);
void kalman_free(KalmanFilter *kf);
void kalman_predict(KalmanFilter *kf, const gsl_vector *u);
void kalman_update(KalmanFilter *kf, const gsl_vector *z);
void kalman_set_system_dynamics_A(KalmanFilter *kf, const gsl_matrix *A);
void kalman_set_system_dynamics_B(KalmanFilter *kf, const gsl_matrix *B);
void kalman_set_observation_model_H(KalmanFilter *kf, const gsl_matrix *H);
void kalman_set_process_noise_covariance_Q(KalmanFilter *kf, const gsl_matrix *Q);
void kalman_set_measurement_noise_covariance_R(KalmanFilter *kf, const gsl_matrix *R);
void kalman_set_initial_state_x0(KalmanFilter *kf, const gsl_vector *x0);
void kalman_set_initial_estimate_error_covariance_P0(KalmanFilter *kf, const gsl_matrix *P0);
void kalman_process(KalmanFilter *kf, const gsl_vector *u, const gsl_vector *z);

#endif // KALMANFILTER_H
