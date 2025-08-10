// img 是一維 int 陣列，長度是 height * width
// Coef 是一維 double 陣列，長度是 length2 * length2 * 16
__declspec(dllexport)
void cubic_coef(double Cubic_Xinv[16][16],
				int Length,
                int width,
				int *img,
               double *Coef)
{
    int i, j, k, m, u, v, count;
    int Length2 = 2 * Length + 1;
    int Gvalue[16];

    for (i = 0; i < Length2; i++) {
        for (j = 0; j < Length2; j++) {
            count = 0;
            for (k = 0; k < 4; k++) {
                for (m = 0; m < 4; m++) {
                    // 一維陣列索引 = row * width + col
                    Gvalue[count] = img[(i + m) * width + (j + k)];
                    count++;
                }
            }

            for (u = 0; u < 16; u++) {
                double sum = 0.0;
                for (v = 0; v < 16; v++) {
                    sum += Cubic_Xinv[u][v] * Gvalue[v];
                }
                // Coef[i][j][u] = sum
                Coef[(i * Length2 + j) * 16 + u] = sum;
            }
        }
    }
}
