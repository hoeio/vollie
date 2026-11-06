// Vollie r10: runs the GGML_CPU_BENCH kernel timing (arch/arm/quants.c) on the host build, 1 thread.
#include <pthread.h>
void ggml_cpu_init(void);
void ggml_cpu_q3k_bench(int cpu);
int main(void) { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); ggml_cpu_init(); ggml_cpu_q3k_bench(-1); return 0; }
