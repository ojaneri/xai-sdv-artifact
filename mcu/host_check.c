/* Host validation of xai_core.c against the `shap` package (ref.json).
 * Prints JSON: margins, TreeSHAP phi, expected value, pass counts, and
 * KernelSHAP/occlusion phi for sample 0 for a sanity comparison.
 *   gcc -O2 -o host_check host_check.c xai_core.c -lm && ./host_check      */
#include <stdio.h>
#include "xai_core.h"
#include "model.h"

int main(void)
{
    float phi[N_FEAT + 1];
    xai_init();
    printf("{\"ev\": %.7g, \"bg_expected\": %.7g, \"margin\": [", treeshap_expected(), bg_expected);
    for (int i = 0; i < N_TEST; i++) printf("%s%.7g", i ? "," : "", predict(test_x[i]));
    printf("], \"phi\": [");
    for (int i = 0; i < N_TEST; i++) {
        treeshap(test_x[i], phi);
        printf("%s[", i ? "," : "");
        for (int j = 0; j < N_FEAT; j++) printf("%s%.7g", j ? "," : "", phi[j]);
        printf("]");
    }
    xai_passes = 0; treeshap(test_x[0], phi);
    printf("], \"passes_tree\": %u", xai_passes);
    xai_passes = 0; occlusion(test_x[0], phi);
    printf(", \"passes_occl\": %u, \"occl0\": [", xai_passes);
    for (int j = 0; j < N_FEAT; j++) printf("%s%.7g", j ? "," : "", phi[j]);
    xai_passes = 0; lime(test_x[0], 5000, phi);
    printf("], \"passes_lime\": %u", xai_passes);
    xai_passes = 0; kernelshap(test_x[0], 2096, phi);
    printf(", \"passes_kernel\": %u, \"kern0\": [", xai_passes);
    for (int j = 0; j < N_FEAT; j++) printf("%s%.7g", j ? "," : "", phi[j]);
    printf("]}\n");
    return 0;
}
