//includes
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "base.h"
#include "csf.h"



//functions
/*idx_t inline cal_modereferecesize(idx_t rank)
{ idx_t Modereference;
  Modereference = RESULT_MEM_REF / rank;
  return Modereference;
}*/

/*void cal_blocksize(idx_t * blocksize, csf_sptensor* reference)
{  idx_t i_number,j_number ;
   idx_t totalnumber;
   
   i_number = reference[0] / (idx_t)PHISICAL_CORE_ROW_NUM;
   j_number = reference[1] / (idx_t)CALCULATE_CORE_NUM;
   
}*/

void quicksort(idx_t*  sortarray, idx_t*  noarray0, idx_t*  noarray1, double* value, idx_t begin, idx_t end)
{  idx_t  i, j;
   idx_t  temp;
   
   
   if(begin < end)
    {
        i = begin + 1;  // 将sortarray[begin]作为基准数，因此从array[begin+1]开始与基准数比较！
        j = end;        // sortarray[end]是数组的最后一位
          
        while(i < j)
        {
            if(sortarray[i] > sortarray[begin])  // 如果比较的数组元素大于基准数，则交换位置。
            {
                temp = sortarray[i];
                sortarray[i] = sortarray[j];
                sortarray[j] = temp;  // 交换所有数
                temp = noarray0[i];
                noarray0[i] = noarray0[j];
                noarray0[j] = temp;
                temp = noarray1[i];
                noarray1[i] = noarray1[j];
                noarray1[j] = temp;  
                temp = value[i];
                value[i] = value[j];
                value[j] = temp;
                j--;
            }
            else
            {
                i++;  // 将数组向后移一位，继续与基准数比较。
            }
        }
 
        /* 跳出while循环后，i = j。
         * 此时数组被分割成两个部分  -->  array[begin+1] ~ array[i-1] < array[begin]
         *                           -->  array[i+1] ~ array[end] > array[begin]
         * 这个时候将数组array分成两个部分，再将array[i]与array[begin]进行比较，决定array[i]的位置。
         * 最后将array[i]与array[begin]交换，进行两个分割部分的排序！以此类推，直到最后i = j不满足条件就退出！
         */
 
        if(sortarray[i] >= sortarray[begin])  // 这里必须要取等“>=”，否则数组元素由相同的值时，会出现错误！
        {
            i--;
        }
               temp = sortarray[i];
               sortarray[i] = sortarray[begin];
               sortarray[begin] = temp;  // 交换所有数
               temp = noarray0[i];
               noarray0[i] = noarray0[begin];
               noarray0[begin] = temp;
               temp = noarray1[i];
               noarray1[i] = noarray1[begin];
               noarray1[begin] = temp;  
               temp = value[i];
               value[i] = value[begin];
               value[begin] = temp; 
        quicksort(sortarray, noarray0, noarray1, value, begin, i);
        quicksort(sortarray, noarray0, noarray1, value, j, end);
    }
}


