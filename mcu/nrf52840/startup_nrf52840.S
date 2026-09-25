/*
 * Startup for pqm4_masked on nRF52840 (Shepherd Nova target v1.3).
 * Bare-metal: no SoftDevice. Follows the MPS2 pattern so the pqm4
 * build system links unchanged. Linked todos: #t-0034
 */

.syntax unified
.cpu cortex-m4
.fpu fpv4-sp-d16
.thumb

.section .isr_vector, "a", %progbits
.word __StackTop
.word Reset_Handler
.word NMI_Handler
.word HardFault_Handler
.word MemManage_Handler
.word BusFault_Handler
.word UsageFault_Handler
.word 0
.word 0
.word 0
.word 0
.word Default_Handler
.word Default_Handler
.word 0
.word Default_Handler
.word SysTick_Handler
.rept 52
.word Default_Handler
.endr

.section .text
.thumb_func
.global Reset_Handler
Reset_Handler:
    /* Enable FPU (CP10/CP11 full access) */
    ldr r0, =0xE000ED88
    ldr r1, [r0]
    orr r1, r1, #(0xF << 20)
    str r1, [r0]
    dsb
    isb

    /* Copy .data from flash to RAM */
    ldr r0, =_sidata
    ldr r1, =__data_start__
    ldr r2, =__data_end__
1:  cmp r1, r2
    bcs 2f
    ldr r3, [r0], #4
    str r3, [r1], #4
    b 1b
2:
    /* Zero .bss */
    ldr r1, =__bss_start__
    ldr r2, =__bss_end__
    movs r3, #0
3:  cmp r1, r2
    bcs 4f
    str r3, [r1], #4
    b 3b
4:
    bl main
hang:
    wfi
    b hang

.weak NMI_Handler
.thumb_set NMI_Handler, Default_Handler
.weak HardFault_Handler
.thumb_set HardFault_Handler, Default_Handler
.weak MemManage_Handler
.thumb_set MemManage_Handler, Default_Handler
.weak BusFault_Handler
.thumb_set BusFault_Handler, Default_Handler
.weak UsageFault_Handler
.thumb_set UsageFault_Handler, Default_Handler
.weak SysTick_Handler
.thumb_set SysTick_Handler, Default_Handler

.global Default_Handler
.thumb_func
Default_Handler:
    b Default_Handler

.end
