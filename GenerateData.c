#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h> // GSL for random number generation
#include <gsl/gsl_randist.h> // GSL for normal distribution
#include <time.h> // for seeding RNG

// System parameters
const double m = 1.0; // Mass
const double k = 1.0; // Spring constant
const double dt = 0.01; // Time step
const int N = 1000; // Number of time steps
const double noise_std_dev = 0.1; // Standard deviation for noise

void generate_data(const char* filename);

int main(int argc, char **argv) {
    const char* filename = "system_data.txt";
    generate_data(filename);
    return 0;
}

void generate_data(const char* filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    // Initialize the random number generator
    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, time(NULL)); // seed RNG

    double x = 0.0; // Initial position
    double v = 0.0; // Initial velocity
    double a; // Acceleration
    double F; // Force

    for (int i = 0; i < N; ++i) {
        F = 1.0; // Example: constant force, you can modify this as needed
        a = F / m - k * x / m;
        v += a * dt;
        x += v * dt;

        // Add noise to position
        double noisy_x = x + gsl_ran_gaussian(r, noise_std_dev);

        fprintf(fp, "%f\n", noisy_x);
    }

    gsl_rng_free(r);
    fclose(fp);
}
