#include <stddef.h>
#include <string.h>
#include "base.h"
#include "io.h"
#include "sptensor.h"
#include "matrixprocess.h"
#include <stdlib.h>
#include <stdio.h>
//#include "timer.h"



/******************************************************************************
 * FILE TYPES
 *****************************************************************************/
struct ftype
{
  char * extension;
  splatt_file_type type;
};

static struct ftype file_extensions[] = {
  { ".tns", SPLATT_FILE_TEXT_COORD },
  { ".coo", SPLATT_FILE_TEXT_COORD },
  { ".bin", SPLATT_FILE_BIN_COORD  },
  { NULL, 0}
};


splatt_file_type get_file_type(
    char const * const fname)
{
  /* find last . in filename */
  char const * const suffix = strrchr(fname, '.');
  if(suffix == NULL) {
    goto NOT_FOUND;
  }

  size_t idx = 0;
  do {
    if(strcmp(suffix, file_extensions[idx].extension) == 0) {
      return file_extensions[idx].type;
    }
  } while(file_extensions[++idx].extension != NULL);


  /* default to text coordinate format */
  NOT_FOUND:
  fprintf(stderr, "SPLATT: extension for '%s' not recognized. "
                  "Defaulting to ASCII coordinate form.\n", fname);
  return SPLATT_FILE_TEXT_COORD;
}


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/
static sptensor_t * p_tt_read_file(
  FILE * fin)
{
  char * ptr = NULL;
  /* first count nnz in tensor */
  idx_t nnz = 0;
  idx_t nmodes = 3;

  idx_t dims[nmodes];
  idx_t offsets[nmodes];
  tt_get_dims(fin, &nmodes, &nnz, dims, offsets);
  char* strs[MAX_NMODES];
  idx_t m = 0;
  /* allocate structures */
  static sptensor_t * tt = NULL;
   tt = (sptensor_t*) malloc(sizeof(sptensor_t));
   //tt->tiled = SPLATT_NOTILE;
   tt->nnz = nnz;
   tt->vals = (double*)malloc(nnz * sizeof(double));
   tt->nmodes = nmodes;
   tt->dims = (idx_t*)malloc(nmodes * sizeof(idx_t));
   tt->ind  = (idx_t**)malloc(nmodes * sizeof(idx_t*));
   for(m=0; m < nmodes; m++) {
     tt->ind[m] = (idx_t*)malloc(nnz * sizeof(idx_t));}
     memcpy(tt->dims, dims, nmodes * sizeof(idx_t));
  char * line = NULL;
  int64_t read;
  size_t len = 0;
  /* fill in tensor data */
  rewind(fin);
  nnz = 0;
  char** ptr2 = NULL;  //for test
  char* token;       //for test
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
     if(read > 1 && line[0] != '#') {
      ptr = line;
      for( m=0; m < nmodes; m++) {
        tt->ind[m][nnz] = strtoul(ptr,&ptr,10);
              }     
         tt->vals[nnz++] = strtod(ptr,&ptr);
        //printf("value:%lf\n",tt->vals[nnz-1]);           
    }
  }
  
  free(line); 
  
  return tt;
}


/**
* @brief Write a binary header to an input file.
*
* @param fout The file to write to.
* @param tt The tensor to form a header from.
* @param[out] header The header to write.
*/
static void p_write_tt_binary_header(
  FILE * fout,
  sptensor_t const * const tt,
  bin_header * header)
{
  int32_t type = SPLATT_BIN_COORD;
  fwrite(&type, sizeof(type), 1, fout);

  /* now see if all indices fit in 32bit values */
  uint64_t idx = tt->nnz < UINT32_MAX ?  sizeof(idx_t) : sizeof(uint64_t);
  idx_t m = 0;
  for(m=0; m < tt->nmodes; ++m) {
    if(tt->dims[m] > UINT32_MAX) {
      idx = sizeof(uint64_t);
      break;
    }
  }

  /* now see if every value can exactly be represented as a float */
  uint64_t val = sizeof(float);
  idx_t n = 0;
  double conv;
  for(n=0; n < tt->nnz; ++n) {
    conv = tt->vals[n];
    if( conv != tt->vals[n]) {
      val = sizeof(double);
    }
  }

  header->magic = type;
  header->idx_width = idx;
  header->val_width = val;

  fwrite(&idx, sizeof(idx), 1, fout);
  fwrite(&val, sizeof(val), 1, fout);
}



/*
* @brief Read a COORD tensor from a binary file, converting from smaller idx or
*        val precision if necessary.
*
* @param fin The file to read from.
*
* @return The parsed tensor.
*/
static sptensor_t * p_tt_read_binary_file(
  FILE * fin)
{
  bin_header header;
  read_binary_header(fin, &header);

  idx_t nnz = 0;
  idx_t nmodes = 0;
  idx_t dims[MAX_NMODES];
  idx_t m = 0; 
 
  fill_binary_idx(&nmodes, 1, &header, fin);
  fill_binary_idx(dims, nmodes, &header, fin);
  fill_binary_idx(&nnz, 1, &header, fin);

  /* allocate structures */
  sptensor_t * tt = tt_alloc(nnz, nmodes);
  memcpy(tt->dims, dims, nmodes * sizeof(*dims));

  /* fill in tensor data */
  for(m=0; m < nmodes; ++m) {
    fill_binary_idx(tt->ind[m], nnz, &header, fin);
  }
  fill_binary_val(tt->vals, nnz, &header, fin);

  return tt;
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
sptensor_t * tt_read_file(
  char const * const fname)
{
  FILE * fin;
  if((fin = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n", fname);
    return NULL;
  }

  static sptensor_t * tt = NULL;

  switch(get_file_type(fname)) {
    case SPLATT_FILE_TEXT_COORD:
      tt = p_tt_read_file(fin);
      break;
    case SPLATT_FILE_BIN_COORD:
      tt = p_tt_read_binary_file(fin);
      break;
  }
  fclose(fin);
  return tt;
}


sptensor_t * tt_read_binary_file(
  char const * const fname)
{
  FILE * fin;
  if((fin = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n", fname);
    return NULL;
  }
  
  sptensor_t * tt = p_tt_read_binary_file(fin);
  
  return tt;
}


void tt_get_dims(
    FILE * fin,
    idx_t* const outnmodes,
    idx_t* const outnnz,
    idx_t* outdims,
    idx_t* offset)
{
  char * ptr = NULL;
  idx_t nnz = 0;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;
  idx_t m = 0;
  idx_t ind;
  /* first count modes in tensor */
  idx_t nmodes = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    if(read > 1 && line[0] != '#') {
      /* get nmodes from first nnz line */
      ptr = strtok(line, " \t");
      while(ptr != NULL) {
        ++nmodes;
        ptr = strtok(NULL, " \t");
      }
      break;
    }
  }
  --nmodes;

  for(m=0; m < nmodes; ++m) {
    outdims[m] = 0;
    offset[m] = 1;
  }

  /* fill in tensor dimensions */
  rewind(fin);
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      ptr = line;
      for(m=0; m < nmodes; ++m) {
         ind = strtoull(ptr, &ptr, 10);

        /* outdim is maximum */
        outdims[m] = (ind > outdims[m]) ? ind : outdims[m];

        /* offset is minimum */
        offset[m] = (ind < offset[m]) ? ind : offset[m];
      }
      /* skip over tensor val */
      strtod(ptr, &ptr);
      ++nnz;
    }
  }
  *outnnz = nnz;
  *outnmodes = nmodes;

  /* only support 0 or 1 indexing */
  for(m=0; m < nmodes; ++m) {
    if(offset[m] != 0 && offset[m] != 1) {
      
      exit(1);
    }
  }

  /* adjust dims when zero-indexing */
  for(m=0; m < nmodes; ++m) {
    if(offset[m] == 0) {
      ++outdims[m];
    }
  }

  rewind(fin);
  free(line);
}


void tt_write(
  sptensor_t const * const tt,
  char const * const fname)
{
  FILE * fout;
  if(fname == NULL) {
    fout = stdout;
  } else {
    if((fout = fopen(fname,"w")) == NULL) {
      fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", fname);
      return;
    }
  }

  tt_write_file(tt, fout);

  if(fname != NULL) {
    fclose(fout);
  }
}

void tt_write_file(
  sptensor_t const * const tt,
  FILE * fout)
{
  idx_t m = 0;
  idx_t n = 0;
  for( n=0; n < tt->nnz; ++n) {
    for(m=0; m < tt->nmodes; ++m) ;
      /* files are 1-indexed instead of 0 */
          
    
  }
  }


void tt_write_binary(
  sptensor_t const * const tt,
  char const * const fname)
{
  FILE * fout;
  if(fname == NULL) {
    fout = stdout;
  } else {
    if((fout = fopen(fname,"w")) == NULL) {
      fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", fname);
      return;
    }
  }

  tt_write_binary_file(tt, fout);

  if(fname != NULL) {
    fclose(fout);
  }
}


void tt_write_binary_file(
  sptensor_t const * const tt,
  FILE * fout)
{
  
  idx_t m = 0;
  idx_t n = 0;
  bin_header header;
  p_write_tt_binary_header(fout, tt, &header);

  /* WRITE INDICES */

  /* if we are writing to the same precision they are stored in, just fwrite */
  if(header.idx_width == sizeof(idx_t)) {
    fwrite(&tt->nmodes, sizeof(tt->nmodes), 1, fout);
    fwrite(tt->dims, sizeof(*tt->dims), tt->nmodes, fout);
    fwrite(&tt->nnz, sizeof(tt->nnz), 1, fout);
    for(m=0; m < tt->nmodes; ++m) {
      fwrite(tt->ind[m], sizeof(*tt->ind[m]), tt->nnz, fout);
    }

  /* otherwise we convert (downwards) element-wise */
  } else if(header.idx_width < sizeof(idx_t)) {
    idx_t buf = tt->nmodes;
    fwrite(&buf, sizeof(buf), 1, fout);
    for(m=0; m < tt->nmodes; ++m) {
      buf = tt->dims[m];
      fwrite(&buf, sizeof(buf), 1, fout);
    }
    buf = tt->nnz;
    fwrite(&buf, sizeof(buf), 1, fout);
    /* write inds */
    for(m=0; m < tt->nmodes; ++m) {
      for(n=0; n < tt->nnz; ++n) {
        buf = tt->ind[m][n];
        fwrite(&buf, sizeof(buf), 1, fout);
      }
    }

  } else {
    /* XXX this should never be reached */
    fprintf(stderr, "SPLATT: the impossible happened, "
                    "idx_width > IDX_TYPEWIDTH.\n");
    abort();
  }

  /* WRITE VALUES */

  if(header.val_width == sizeof(double)) {
    fwrite(tt->vals, sizeof(*tt->vals), tt->nnz, fout);
  /* otherwise we convert (downwards) element-wise */
  } else if(header.val_width < sizeof(double)) {
    for(n=0; n < tt->nnz; ++n) {
      float buf = tt->vals[n];
      fwrite(&buf, sizeof(buf), 1, fout);
    }

  } else {
    /* XXX this should never be reached */
    fprintf(stderr, "SPLATT: the impossible happened, "
                    "val_width > VAL_TYPEWIDTH.\n");
    abort();
  }

  }


void read_binary_header(
  FILE * fin,
  bin_header * header)
{
  fread(&(header->magic), sizeof(header->magic), 1, fin);
  fread(&(header->idx_width), sizeof(header->idx_width), 1, fin);
  fread(&(header->val_width), sizeof(header->val_width), 1, fin);

  if(header->idx_width > 4) {
    fprintf(stderr, "SPLATT: ERROR input has %zu-bit integers. "
                    "Build with SPLATT_IDX_TYPEWIDTH %zu\n",
                    header->idx_width * 8, header->idx_width * 8);
    exit(EXIT_FAILURE);
  }

  if(header->val_width > 8) {
    fprintf(stderr, "SPLATT: WARNING input has %zu-bit floating-point values. "
                    "Build with SPLATT_VAL_TYPEWIDTH %zu for full precision\n",
                    header->val_width * 8, header->val_width * 8);
  }
}


void fill_binary_idx(
    idx_t * const buffer,
    idx_t const count,
    bin_header const * const header,
    FILE * fin)
{  idx_t i = 0;
  if(header->idx_width == sizeof(idx_t)) {
    fread(buffer, sizeof(idx_t), count, fin);
  } else {

    /* read in idx_t in a buffered fashion */
    idx_t const BUF_LEN = 1024*1024;
    idx_t * ubuf = malloc(BUF_LEN * sizeof(*ubuf));
    idx_t n = 0;
    idx_t const read_count;
    idx_t i = 0;
    for(n=0; n < count; n += BUF_LEN) {
      fread(ubuf, sizeof(*ubuf), read_count, fin);
      for( i=0; i < read_count; ++i) {  //maybe paralleled
        buffer[n + i] = ubuf[i];
      }
    }
    free(ubuf);
  }
}


void fill_binary_val(
    double* const buffer,
    idx_t const count,
    bin_header const * const header,
    FILE * fin)
{ idx_t n = 0;
  idx_t i = 0;
  idx_t read_count;
  if(header->val_width == sizeof(double)) {
    fread(buffer, sizeof(double), count, fin);
  } else {
    /* read in float in a buffered fashion */
    idx_t const BUF_LEN = 1024*1024;

    /* select whichever SPLATT *is not* configured with. */

    float * ubuf = malloc(BUF_LEN * sizeof(*ubuf));

    for(n=0; n < count; n += BUF_LEN) {
      read_count = SS_MIN(BUF_LEN, count - n);
      fread(ubuf, sizeof(*ubuf), read_count, fin);
      for(i=0; i < read_count; ++i) {  //maybe paralleled
        buffer[n + i] = ubuf[i];
      }
    }
    free(ubuf);
  }
}




void spmat_write(
  spordi_matrix const * const mat,
  char const * const fname)
{
  FILE * fout;
  if(fname == NULL || strcmp(fname, "-") == 0) {
    fout = stdout;
  } else {
    if((fout = fopen(fname,"w")) == NULL) {
      fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", fname);
      return;
    }
  }

  spmat_write_file(mat, fout);

  if(fout != stdout) {
    fclose(fout);
  }
}

void spmat_write_file(
  spordi_matrix const * const mat,
  FILE * fout)
{ idx_t i = 0;
  idx_t j = 0;
  /* write CSR matrix */
  for( i=0; i < mat->I; ++i) {
    for(j=mat->rowptr[i]; j < mat->rowptr[i+1]; ++j) {
      
    }
    fprintf(fout, "\n");
  }
 
}

void mat_write(
  ordi_matrix const * const mat,
  char const * const fname)
{
  FILE * fout;
  if(fname == NULL) {
    fout = stdout;
  } else {
    if((fout = fopen(fname,"w")) == NULL) {
      fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", fname);
      return;
    }
  }

  mat_write_file(mat, fout);

  if(fout != stdout) {
    fclose(fout);
  }
}

void mat_write_file(
  ordi_matrix const * const mat,
  FILE * fout)
{
  idx_t const I = mat->I;
  idx_t const J = mat->J;
  double const * const vals = mat->values;
  idx_t i = 0;
  idx_t j = 0;
  if(mat->row_col_flag) {
    for(i=0; i < mat->I; ++i) {
      for(j=0; j < J; ++j) {
        fprintf(fout, "%+0.8le ", vals[j + (i*J)]);
      }
      fprintf(fout, "\n");
    }
  } else {
    for(i=0; i < mat->I; ++i) {
      for(j=0; j < J; ++j) {
        fprintf(fout, "%+0.8le ", vals[i + (j*I)]);
      }
      fprintf(fout, "\n");
    }
  }
  
}


void vec_write(
  double const * const vec,
  idx_t const len,
  char const * const fname)
{
  FILE * fout;
  if(fname == NULL) {
    fout = stdout;
  } else {
    if((fout = fopen(fname,"w")) == NULL) {
      fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", fname);
      return;
    }
  }

  vec_write_file(vec, len, fout);

  if(fout != stdout) {
    fclose(fout);
  }
}

void vec_write_file(
  double const * const vec,
  idx_t const len,
  FILE * fout)
{ idx_t i = 0;
  for(i=0; i < len; ++i) {
    fprintf(fout, "%le\n", vec[i]);
  }

  
}




