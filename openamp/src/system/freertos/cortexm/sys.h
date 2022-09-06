/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * @file	freertos/cortexm/sys.h
 * @brief	cortexm system primitives for libmetal.
 */

#ifndef __METAL_FREERTOS_SYS__H__
#error "Include metal/freertos/sys.h instead of metal/freertos/cortexm/sys.h"
#endif

#ifndef __METAL_FREERTOS_CORTEXM_SYS__H__
#define __METAL_FREERTOS_CORTEXM_SYS__H__

#ifdef __cplusplus
extern "C" {
#endif

#ifdef METAL_INTERNAL

void sys_irq_enable(unsigned int vector);

void sys_irq_disable(unsigned int vector);

#endif /* METAL_INTERNAL */

#ifdef __cplusplus
}
#endif

#endif /* __METAL_FREERTOS_CORTEXM_SYS__H__ */
