/*
 * Copyright (c) 2020 STMicroelectronnics. All rights reserved.
 * SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * @file      metal/errno.h
 * @brief     error specific primitives for libmetal.
 */

#ifndef __METAL_ERRNO__H__
#define __METAL_ERRNO__H__

#if defined(__ICCARM__)
# include <metal/compiler/iar/errno.h>
#elif defined(__CC_ARM) || defined (__ARMCC_VERSION)
# include <metal/compiler/armcc/errno.h>
#else
# include <errno.h>
#endif

#endif /* __METAL_ERRNO__H__ */
