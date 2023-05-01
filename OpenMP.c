#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "pgmio.h"

double start_time_sobel, end_time_sobel;
double start_time_total, end_time_total;

#define M 256
#define N 256
#define THRESH 100

int main(int argc, char **argv)
{
    int i, j;
    float image[M][N];
    float masterbuf[M][N];

    char *filename;
    char input[] = "image256x256.pgm";
    char output[] = "image-output256x256.pgm";
    filename = input;

    start_time_total = omp_get_wtime();
    pgmread(filename, masterbuf, M, N);

    printf("width: %d \nheight: %d\n", M, N);
    start_time_sobel = omp_get_wtime();

    omp_set_num_threads(4); // set the number of threads to 4

    #pragma omp parallel shared(image, masterbuf)
    {
        #pragma omp for schedule(static)
        for (i = 1; i < M - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                float gradient_h = ((-1.0 * masterbuf[i - 1][j - 1]) + (1.0 * masterbuf[i + 1][j - 1]) + (-2.0 * masterbuf[i - 1][j]) + (2.0 * masterbuf[i + 1][j]) + (-1.0 * masterbuf[i - 1][j + 1]) + (1.0 * masterbuf[i + 1][j + 1]));
                float gradient_v = ((-1.0 * masterbuf[i - 1][j - 1]) + (-2.0 * masterbuf[i][j - 1]) + (-1.0 * masterbuf[i + 1][j - 1]) + (1.0 * masterbuf[i - 1][j + 1]) + (2.0 * masterbuf[i][j + 1]) + (1.0 * masterbuf[i + 1][j + 1]));
                float gradient = sqrt((gradient_h * gradient_h) + (gradient_v * gradient_v));
                if (gradient < THRESH) {
                    gradient = 0;
                }
                else {
                    gradient = 255;
                }
                image[i][j] = gradient;
            }
        }
    }

    end_time_sobel = omp_get_wtime();

    printf("Finished\n");

    filename = output;
    printf("Output: <%s>\n", filename);
    pgmwrite(filename, image, M, N);

    end_time_total = omp_get_wtime();

    double total = (end_time_sobel - start_time_sobel);
    printf("Total Parallel Time: %fs\n", total);
    printf("Total Serial Time: %fs\n", (end_time_total - start_time_total) - total);
    printf("Total Time: %fs\n", end_time_total - start_time_total);

    return 0;
}
