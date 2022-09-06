/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * @file	freertos/cortexm/sys.c
 * @brief	cortex m system primitives implementation.
 */

#include <metal/io.h>
#include <metal/sys.h>
#include <metal/utilities.h>
#include <stdint.h>

void sys_irq_restore_enable(unsigned int flags)
{
	metal_unused(flags);
	/* we disable/enable all IRQs */
	__enable_irq();
}

unsigned int sys_irq_save_disable(void)
{
	/* we disable/enable all IRQs */
	__disable_irq();
	return 0;
}

void sys_irq_enable(unsigned int vector)
{
	NVIC_EnableIRQ(vector);
}

void sys_irq_disable(unsigned int vector)
{
	NVIC_DisableIRQ(vector);
}

void metal_machine_cache_flush(void *addr, unsigned int len)
{
#if (defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U))
	SCB_CleanDCache_by_Addr(addr, len);
#else
	metal_unused(addr);
	metal_unused(len);
#endif
}

void metal_machine_cache_invalidate(void *addr, unsigned int len)
{
#if (defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U))
	SCB_InvalidateDCache_by_Addr(addr, len);
#else
	metal_unused(addr);
	metal_unused(len);
#endif
}

void *metal_machine_io_mem_map(void *va, metal_phys_addr_t pa,
			       size_t size, unsigned int flags)
{
	metal_unused(pa);
	metal_unused(size);
	metal_unused(flags);

	return va;
}
