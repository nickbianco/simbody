/*
 * -----------------------------------------------------------------
 * $Revision: 1.2 $
 * $Date: 2006/11/29 00:05:06 $
 * ----------------------------------------------------------------- 
 * Programmer: Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * Copyright (c) 2006, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 * For details, see the LICENSE file.
 * -----------------------------------------------------------------
 * This is the header file for the CPODES scaled preconditioned 
 * Bi-CGSTAB linear solver, CPSPBCG.
 * -----------------------------------------------------------------
 */

#ifndef _CPSPBCG_H
#define _CPSPBCG_H

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

#include <cpodes/cpodes_spils.h>
#include <sundials/sundials_spbcgs.h>

/*
 * -----------------------------------------------------------------
 * Function : CPSpbcg
 * -----------------------------------------------------------------
 * A call to the CPSpbcg function links the main CPODES integrator
 * with the CPSPBCG linear solver.
 *
 * cpode_mem is the pointer to the integrator memory returned by
 *           CPodeCreate.
 *
 * pretype   is the type of user preconditioning to be done.
 *           This must be one of the four enumeration constants
 *           PREC_NONE, PREC_LEFT, PREC_RIGHT, or PREC_BOTH defined
 *           in iterative.h. These correspond to no preconditioning,
 *           left preconditioning only, right preconditioning
 *           only, and both left and right preconditioning,
 *           respectively.
 *
 * maxl      is the maximum Krylov dimension. This is an
 *           optional input to the CPSPBCG solver. Pass 0 to
 *           use the default value CPSPBCG_MAXL=5.
 *
 * The return value of CPSpbcg is one of:
 *    CPSPILS_SUCCESS   if successful
 *    CPSPILS_MEM_NULL  if the CPODES memory was NULL
 *    CPSPILS_MEM_FAIL  if there was a memory allocation failure
 *    CPSPILS_ILL_INPUT if a required vector operation is missing
 * The above constants are defined in cpodes_spils.h
 *
 * -----------------------------------------------------------------
 */

SUNDIALS_EXPORT int CPSpbcg(void *cpode_mem, int pretype, int maxl);


#ifdef __cplusplus
}
#endif

#endif
