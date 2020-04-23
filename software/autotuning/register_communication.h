#ifndef REGISTER_COMMUNICATION_H
#define REGISTER_COMMUNICATION_H

#include <math.h>
#include "slave.h"
#include "simd.h"

#define COL(x) (x & 0x07)
#define ROW(x) ((x & 0x38) >> 3)
#define REG_PUTR(var, dest) asm volatile ("putr %0,%1\n"::"r"(var),"r"(dest))
#define REG_PUTC(var, dest) asm volatile ("putc %0,%1\n"::"r"(var),"r"(dest))
#define REG_GETR(var) asm volatile ("getr %0\n":"=r"(var))
#define REG_GETC(var) asm volatile ("getc %0\n":"=r"(var))
#define ROWSYN() athread_syn(ROW_SCOPE,0xff)
#define COLSYN() athread_syn(COL_SCOPE,0xff)
#define ALLSYN() athread_syn(ARRAY_SCOPE,0xffff)

#define MEMORY() asm volatile ("":::"memory")

#endif
