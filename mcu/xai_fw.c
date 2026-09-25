/*
 * XAI cost firmware: nRF52840 (Shepherd Nova, bare metal) or STM32L4 via the
 * pqm4 libopencm3 HAL (-DXAI_OCM3: NUCLEO-L476RG, st-iotnode).
 *
 * Runs the explanation methods of xai_core.c in gated blocks: the gate pin
 * (P0.04 = Shepherd GPIO2) is high exactly while a block runs, so the
 * calibrated V/I trace gives the energy of each block and the UART line that
 * follows gives its cycle count and pass count. Blocks repeat in rounds
 * until the experiment ends.
 *
 * Register values are those of pqm4_masked/common/hal-nrf52840.c, validated
 * on the lab by probe_v3 (experiment dad58964, 2026-09-14). No interrupts,
 * no SysTick: firmwares that used SysTick+WFI did not run on this lab.
 *
 * UART line format (115200 8N1):
 *   B r=<round> op=<name> reps=<R> passes=<total passes> cyc=<total cycles>
 */
#include <stdint.h>
#include "xai_core.h"
#include "model.h"

#if defined(XAI_OCM3)
/* ---------- STM32L4 (NUCLEO-L476RG, st-iotnode) via pqm4 libopencm3 HAL ----
 * CLOCK_BENCHMARK = HSI16, 16 MHz, 0 flash wait states (pqm4 convention).
 * Time is SysTick with an overflow interrupt (hal_get_time). No gate pin:
 * these boards give cycles only. Output goes through hal_send_str.        */
#include <hal.h>
#define CPU_HZ   16000000UL
#ifndef ROUNDS
#define ROUNDS   2
#endif
static uint64_t now(void) { return hal_get_time(); }
static void tick(void) {}
static void gate_on(void) {}
static void gate_off(void) {}
static char line[512];
static int llen;
static void putc_(char c)
{
    if (c == '\r') return;
    if (c == '\n') { line[llen] = 0; hal_send_str(line); llen = 0; return; }
    if (llen < (int)sizeof line - 1) line[llen++] = c;
}
static void setup(void) { hal_setup(CLOCK_BENCHMARK); }
#else
/* ---------- nRF52840, Shepherd Nova target v1.3, bare metal ------------- */
#define NRF_GPIO_BASE      0x50000000UL
#define NRF_UART0_BASE     0x40002000UL
#define GPIO_OUTSET        (*(volatile uint32_t *)(NRF_GPIO_BASE + 0x508))
#define GPIO_OUTCLR        (*(volatile uint32_t *)(NRF_GPIO_BASE + 0x50C))
#define GPIO_PIN_CNF(n)    (*(volatile uint32_t *)(NRF_GPIO_BASE + 0x700 + (n)*4))
#define UART_TASKS_STARTTX (*(volatile uint32_t *)(NRF_UART0_BASE + 0x008))
#define UART_EVENTS_TXDRDY (*(volatile uint32_t *)(NRF_UART0_BASE + 0x11C))
#define UART_TXD           (*(volatile uint32_t *)(NRF_UART0_BASE + 0x51C))
#define UART_BAUDRATE      (*(volatile uint32_t *)(NRF_UART0_BASE + 0x524))
#define UART_CONFIG        (*(volatile uint32_t *)(NRF_UART0_BASE + 0x56C))
#define UART_PSEL_TXD      (*(volatile uint32_t *)(NRF_UART0_BASE + 0x50C))
#define UART_PSEL_RXD      (*(volatile uint32_t *)(NRF_UART0_BASE + 0x514))
#define UART_ENABLE        (*(volatile uint32_t *)(NRF_UART0_BASE + 0x500))
#define DWT_CTRL           (*(volatile uint32_t *)0xE0001000)
#define DWT_CYCCNT         (*(volatile uint32_t *)0xE0001004)
#define SCB_DEMCR          (*(volatile uint32_t *)0xE000EDFC)

#define PIN_UART_TX 8
#define PIN_UART_RX 21
#define PIN_GATE    4
#define CPU_HZ      64000000UL
#ifndef ROUNDS
#define ROUNDS      0                    /* 0 = until the experiment ends */
#endif

/* 64-bit cycle clock: DWT wraps every 67 s at 64 MHz, so every long loop
 * calls now() through xai_tick often enough to see each wrap. */
static uint32_t cyc_last, cyc_high;
static uint64_t now(void)
{
    uint32_t c = DWT_CYCCNT;
    if (c < cyc_last) cyc_high++;
    cyc_last = c;
    return ((uint64_t)cyc_high << 32) | c;
}
static void tick(void) { (void)now(); }
static void gate_on(void) { GPIO_OUTSET = 1U << PIN_GATE; }
static void gate_off(void) { GPIO_OUTCLR = 1U << PIN_GATE; }

static void putc_(char c)
{
    UART_EVENTS_TXDRDY = 0;
    UART_TXD = (uint8_t)c;
    while (!UART_EVENTS_TXDRDY) {}
}

static void setup(void)
{
    GPIO_PIN_CNF(PIN_UART_TX) = 1;
    GPIO_PIN_CNF(PIN_UART_RX) = 0;
    GPIO_PIN_CNF(PIN_GATE) = 1;
    GPIO_OUTCLR = 1U << PIN_GATE;
    UART_PSEL_TXD = PIN_UART_TX;
    UART_PSEL_RXD = PIN_UART_RX;
    UART_BAUDRATE = 0x01D7E000;          /* 115200 */
    UART_CONFIG = 0;
    UART_ENABLE = 4;
    UART_TASKS_STARTTX = 1;
    SCB_DEMCR |= 1U << 24;
    DWT_CTRL |= 1;
    DWT_CYCCNT = 0;
}
#endif

static void puts_(const char *s) { while (*s) putc_(*s++); }
static void putu(uint64_t v)
{
    char b[21]; int i = 20; b[i] = 0;
    do { b[--i] = (char)('0' + v % 10); v /= 10; } while (v);
    puts_(b + i);
}
static void puti(int64_t v) { if (v < 0) { putc_('-'); v = -v; } putu((uint64_t)v); }
static void putf6(float f) { puti((int64_t)(f * 1e6f + (f >= 0 ? 0.5f : -0.5f))); } /* x1e6 */

static void spin(uint64_t cycles)
{
    uint64_t t0 = now();
    while (now() - t0 < cycles) {}
}

static float phi[N_FEAT + 1];
static volatile float sink;

static void block(int round, const char *op, int reps, int kind, int arg)
{
    xai_seed(20260823u + (uint32_t)round);
    xai_passes = 0;
    gate_on();
    uint64_t t0 = now();
    for (int r = 0; r < reps; r++) {
        const float *x = test_x[r % N_TEST];
        switch (kind) {
        case 0: spin((uint64_t)arg); break;
        case 1: sink = predict(x); break;
        case 2: sink = predict(bg_x[r % N_BG]); break;
        case 3: treeshap(x, phi); break;
        case 4: occlusion(x, phi); break;
        case 5: lime(x, arg, phi); break;
        case 6: kernelshap(x, arg, phi); break;
        }
        tick();
    }
    uint64_t t1 = now();
    gate_off();
    puts_("B r="); putu((uint64_t)round);
    puts_(" op="); puts_(op);
    puts_(" reps="); putu((uint64_t)reps);
    puts_(" passes="); putu(xai_passes);
    puts_(" cyc="); putu(t1 - t0);
    puts_("\r\n");
    spin(CPU_HZ / 4);                    /* 250 ms gate-low gap */
}

int main(void)
{
    setup();
    spin(CPU_HZ * 2);                    /* level-converter transient, ~1 s */
    xai_tick = tick;
    xai_init();
    puts_("xai-fw boot nodes="); putu(N_NODES);
    puts_(" trees="); putu(N_TREES); puts_(" hz="); putu(CPU_HZ); puts_("\r\n");

    /* correctness: margins and TreeSHAP of every test sample, x1e6 */
    for (int i = 0; i < N_TEST; i++) {
        treeshap(test_x[i], phi);
        puts_("C i="); putu((uint64_t)i);
        puts_(" m="); putf6(predict(test_x[i]));
        puts_(" phi=");
        for (int j = 0; j < N_FEAT; j++) { if (j) putc_(','); putf6(phi[j]); }
        puts_("\r\n");
    }
    puts_("C ev="); putf6(treeshap_expected()); puts_(" bgE="); putf6(bg_expected); puts_("\r\n");

    for (int round = 0; ROUNDS == 0 || round < ROUNDS; round++) {
        block(round, "spin1s", 1, 0, (int)CPU_HZ);
        block(round, "infer_test", 2000, 1, 0);
        block(round, "infer_bg", 1000, 2, 0);
        block(round, "treeshap", 200, 3, 0);
        block(round, "occlusion", 200, 4, 0);
        block(round, "lime5000", 3, 5, 5000);
        block(round, "kernel21", 3, 6, 21);
        block(round, "kernel210", 1, 6, 210);
        block(round, "kernel2096", 1, 6, 2096);
    }
    puts_("#\r\n");                    /* end marker for capture.py */
    for (;;) {}
}
