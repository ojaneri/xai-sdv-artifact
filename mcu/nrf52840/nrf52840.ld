/*
 * Linker script for pqm4_masked on nRF52840 (Shepherd Nova target v1.3).
 *
 * Memory map: 1 MB flash at 0x0, 256 KB RAM at 0x20000000.
 * Bare-metal, no SoftDevice — flash starts at address 0.
 *
 * Follows the MPS2 pattern (common/mps2/MPS2.ld) so the pqm4 build system
 * links it unchanged: entry Reset_Handler, symbols `end` for _sbrk and
 * `__StackTop` for the vector table.
 *
 * Linked todos: #t-0034
 */

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 1024K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 256K
}

ENTRY(Reset_Handler)

_stack_size = 64K; /* pqm4_masked Kyber768 s=4 needs a large stack */

SECTIONS
{
  .isr_vector :
  {
    __vector_table = .;
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } > FLASH

  .text :
  {
    . = ALIGN(4);
    *(.text*)
    *(.rodata*)
    KEEP(*(.init))
    KEEP(*(.fini))
    . = ALIGN(4);
    _etext = .;
    __etext = .;
  } > FLASH

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > FLASH

  __exidx_start = .;
  .ARM.exidx :
  {
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  } > FLASH
  __exidx_end = .;

  _ldata = ALIGN(__exidx_end, 4); /* load .data AFTER .ARM.exidx: avoids LMA overlap */

  .data : AT(_ldata)
  {
    . = ALIGN(4);
    __data_start__ = .;
    *(.data*)
    . = ALIGN(4);
    __preinit_array_start = .;
    KEEP(*(.preinit_array))
    __preinit_array_end = .;
    . = ALIGN(4);
    __init_array_start = .;
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    __init_array_end = .;
    . = ALIGN(4);
    __fini_array_start = .;
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    __fini_array_end = .;
    . = ALIGN(4);
    __data_end__ = .;
  } > RAM
  _sidata = LOADADDR(.data);

  .bss (NOLOAD) :
  {
    . = ALIGN(4);
    __bss_start__ = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > RAM

  /* Heap: from end of .bss to stack limit, consumed via _sbrk */
  .heap (NOLOAD) :
  {
    . = ALIGN(8);
    __end__ = .;
    end = .;
    __HeapBase = .;
    . = ORIGIN(RAM) + LENGTH(RAM) - _stack_size;
    __HeapLimit = .;
  } > RAM

  __StackTop = ORIGIN(RAM) + LENGTH(RAM);
  __StackLimit = __StackTop - _stack_size;
  PROVIDE(__stack = __StackTop);

  ASSERT(__bss_end__ <= __HeapLimit, "RAM overflowed: bss exceeds RAM minus stack")
}