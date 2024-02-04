#include "LQR.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_complex_math.h>

// Initialize LQR controller
void lqr_init(LQR *lqr, int state_size, int control_size) {
    lqr->A = gsl_matrix_alloc(state_size, state_size);
    lqr->B = gsl_matrix_alloc(state_size, control_size);
    lqr->Q = gsl_matrix_alloc(state_size, state_size);
    lqr->R = gsl_matrix_alloc(control_size, control_size);
    lqr->K = gsl_matrix_alloc(control_size, state_size);
    // Set cost matrices for LQR controller
    gsl_matrix_set(lqr->Q, 0, 0, 1.0); // State cost
    gsl_matrix_set(lqr->R, 0, 0, 0.01); // Control effort cost
}


// Function to check if an eigenvalue is stable (negative real part)
int is_stable_eigenvalue(gsl_complex eval) {
    return GSL_REAL(eval) < 0.0;
}

// Function to solve the Discrete-time Algebraic Riccati Equation (DARE)
void solve_dare(const gsl_matrix *A, const gsl_matrix *B, const gsl_matrix *Q, const gsl_matrix *R, gsl_matrix *P) {
    size_t n = A->size1;  // Dimension of the state vector

    // Construct the Hamiltonian matrix H
    gsl_matrix *H = gsl_matrix_calloc(2 * n, 2 * n);
    gsl_matrix_view H_tl = gsl_matrix_submatrix(H, 0, 0, n, n);
    gsl_matrix_view H_tr = gsl_matrix_submatrix(H, 0, n, n, n);
    gsl_matrix_view H_bl = gsl_matrix_submatrix(H, n, 0, n, n);
    gsl_matrix_view H_br = gsl_matrix_submatrix(H, n, n, n, n);

    gsl_matrix_memcpy(&H_tl.matrix, A);
    gsl_matrix_memcpy(&H_br.matrix, A);
    gsl_matrix_scale(&H_br.matrix, -1.0);
    gsl_matrix_transpose_memcpy(&H_br.matrix, &H_br.matrix);

    gsl_matrix *BRBt = gsl_matrix_alloc(n, n);
    gsl_matrix *R_inv = gsl_matrix_alloc(B->size2, B->size2);

    gsl_matrix_memcpy(R_inv, R);
    gsl_linalg_cholesky_decomp(R_inv);
    gsl_linalg_cholesky_invert(R_inv);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, B, R_inv, 0.0, BRBt);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, BRBt, B, 0.0, &H_tr.matrix);

    gsl_matrix_memcpy(&H_bl.matrix, Q);
    gsl_matrix_scale(&H_bl.matrix, -1.0);

    // Solve for eigenvalues and eigenvectors of the Hamiltonian matrix
    gsl_eigen_nonsymmv_workspace *w = gsl_eigen_nonsymmv_alloc(2 * n);
    gsl_vector_complex *eval = gsl_vector_complex_alloc(2 * n);
    gsl_matrix_complex *evec = gsl_matrix_complex_alloc(2 * n, 2 * n);

    gsl_eigen_nonsymmv(H, eval, evec, w);

    // Find stable eigenvectors and reconstruct P
    gsl_matrix *V = gsl_matrix_alloc(n, n);
    int stable_idx = 0;

    for (size_t i = 0; i < 2 * n; ++i) {
        gsl_complex eval_i = gsl_vector_complex_get(eval, i);
        if (GSL_REAL(eval_i) < 0) {  // Stable eigenvalue
            for (size_t j = 0; j < n; ++j) {
                gsl_complex evec_ij = gsl_matrix_complex_get(evec, j + n, i);  // Bottom half
                gsl_matrix_set(V, j, stable_idx, GSL_REAL(evec_ij));
            }
            stable_idx++;
            if (stable_idx == n) break;  // Found enough stable eigenvectors
        }
    }

    gsl_matrix *V_inv = gsl_matrix_alloc(n, n);
    gsl_permutation *perm = gsl_permutation_alloc(n);
    int s;

    gsl_linalg_LU_decomp(V, perm, &s);
    gsl_linalg_LU_invert(V, perm, V_inv);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, V, V_inv, 0.0, P);

    // Cleanup
    gsl_matrix_free(H);
    gsl_matrix_free(BRBt);
    gsl_matrix_free(R_inv);
    gsl_matrix_free(V);
    gsl_matrix_free(V_inv);
    gsl_permutation_free(perm);
    gsl_eigen_nonsymmv_free(w);
    gsl_vector_complex_free(eval);
    gsl_matrix_complex_free(evec);
}



// Free resources allocated for LQR controller
void lqr_free(LQR *lqr) {
    gsl_matrix_free(lqr->A);
    gsl_matrix_free(lqr->B);
    gsl_matrix_free(lqr->Q);
    gsl_matrix_free(lqr->R);
    gsl_matrix_free(lqr->P);
    gsl_matrix_free(lqr->K);
}

// Compute the LQR gain matrix K
void lqr_compute_gain(LQR *lqr) {
    // Solve DARE for P
    solve_dare(lqr->A, lqr->B, lqr->Q, lqr->R, lqr->P);

    // Compute feedback gain K = (R + B^T P B)^{-1} B^T P A
    gsl_matrix *BTP = gsl_matrix_alloc(lqr->B->size2, lqr->P->size2);
    gsl_matrix *BTPB = gsl_matrix_alloc(lqr->B->size2, lqr->B->size2);
    gsl_matrix *BTPB_R = gsl_matrix_alloc(lqr->R->size1, lqr->R->size2);
    gsl_matrix *BTPB_R_inv = gsl_matrix_alloc(lqr->R->size1, lqr->R->size2);
    gsl_matrix *BTPA = gsl_matrix_alloc(lqr->B->size2, lqr->A->size2);
    gsl_matrix *K_temp = gsl_matrix_alloc(lqr->K->size1, lqr->K->size2);

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, lqr->B, lqr->P, 0.0, BTP);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, BTP, lqr->B, 0.0, BTPB);
    gsl_matrix_memcpy(BTPB_R, lqr->R);
    gsl_matrix_add(BTPB_R, BTPB);
    gsl_permutation *perm = gsl_permutation_alloc(BTPB_R->size1);
    int s;
    gsl_linalg_LU_decomp(BTPB_R, perm, &s);
    gsl_linalg_LU_invert(BTPB_R, perm, BTPB_R_inv);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, BTP, lqr->A, 0.0, BTPA);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, BTPB_R_inv, BTPA, 0.0, K_temp);
    gsl_matrix_memcpy(lqr->K, K_temp);

    // Free allocated memory
    gsl_matrix_free(BTP);
    gsl_matrix_free(BTPB);
    gsl_matrix_free(BTPB_R);
    gsl_matrix_free(BTPB_R_inv);
    gsl_matrix_free(BTPA);
    gsl_matrix_free(K_temp);
    gsl_permutation_free(perm);
}

// Set the system dynamics matrix A
void lqr_set_system_dynamics_A(LQR *lqr, const gsl_matrix *A) {
    gsl_matrix_memcpy(lqr->A, A);
}

// Set the control input matrix B
void lqr_set_system_dynamics_B(LQR *lqr, const gsl_matrix *B) {
    gsl_matrix_memcpy(lqr->B, B);
}

// Set the state cost matrix Q
void lqr_set_cSost_matrix_Q(LQR *lqr, const gsl_matrix *Q) {
    gsl_matrix_memcpy(lqr->Q, Q);
}

// Set the control input cost matrix R
void lqr_set_cost_matrix_R(LQR *lqr, const gsl_matrix *R) {
    gsl_matrix_memcpy(lqr->R, R);
}

// Compute the control input based on the current state
void lqr_compute_control_input(LQR *lqr, const gsl_vector *x, gsl_vector *u) {
    // u = -Kx
    gsl_blas_dgemv(CblasNoTrans, -1.0, lqr->K, x, 0.0, u);
}

// Function to encapsulate the computation of LQR gain and control input
void lqr_process(LQR *lqr, const gsl_vector *x, gsl_vector *u) {
    // Ensure the LQR gain K is computed based on the current system matrices
    lqr_compute_gain(lqr);

    // Compute the control input based on the current state and the LQR gain
    lqr_compute_control_input(lqr, x, u);
}
