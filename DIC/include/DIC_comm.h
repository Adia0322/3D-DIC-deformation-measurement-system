#ifndef DIC_COMM_H
#define DIC_COMM_H
#include <stdlib.h>
#include <stdint.h>
#include "PSO_comm.h"

#define square(x) ((x)*(x))

typedef struct DIC_config{
    uint16_t subset_side_len;
    uint16_t scan_side_len;
} DIC_config;

typedef struct DIC_context {
    struct DIC_config config;
    double *img_ref_pt_pos;
    double *img_cur_pt_pos;
    void *priv;
} DIC_context;

typedef struct DIC_subset_info {
    double *subset_data;
    double mean;
    double ref_sum_den;
    int side_len;
    int total_pixels;
} DIC_subset_info;

typedef struct DIC_ZNCC_context {
    struct DIC_subset_info ref_subset_info;
    struct DIC_subset_info cur_subset_info;
    double subset_pt_cur_pos[2];                 // Pi_x, Pi_y relative correct coordinate
    // double subset_pt_ref_pos[2];
    double img_pt_ref_pos[2];                      // object point
    double img_pt_cur_pos[2];
    double *temp_subset;
    double *ref_subset_zero_mean;
    int margin;
} DIC_ZNCC_context;

typedef struct DIC_SSD_context {
    struct DIC_subset_info ref_subset_info;
    struct DIC_subset_info cur_subset_info;
    double subset_pt_cur_pos[2]; 
    double img_pt_ref_pos[2]; 
}DIC_SSD_context;

struct PSO_cost_func_ops {
    int (*init)(SYS_INFO *info);
    double (*cost_function)(PSO_context *ctx);
    int (*cleanup)(PSO_context *ctx);
};

double get_subset_mean(const DIC_subset_info *subset);
void ZNCC_ctx_init(struct SYS_INFO *info);
int ZNCC_subset_alloc(struct DIC_ZNCC_context *zncc_ctx);
int ZNCC_init(struct SYS_INFO *info);
double ZNCC_cost_function(struct PSO_context *ctx) ;
int ZNCC_cleanup(struct PSO_context *ctx);
int SSD_init(struct SYS_INFO *info);
double SSD_cost_function(struct PSO_context *ctx);
int SSD_cleanup(struct PSO_context *ctx);

extern void ZNCC_ctx_init(struct SYS_INFO *info);
extern struct PSO_cost_func_ops zncc_ops;
extern struct PSO_cost_func_ops ssd_ops;

/*
private 為特定任務中所需的資料 (pointer: 8bytes)
*/

#endif
