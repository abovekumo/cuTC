#include <stdio.h>
#include <math.h>
//#include <malloc.h>
#include "slave.h"
#include "common.h"

int ini_flag=0,n_flag,func_flag;
double *OShift,*M,*y,*z,*x_bound;

void sphere_func (double *, double *, int , double *, int); /* Sphere */
void dif_powers_func(double *, double *, int , double *, int);  /* Different Powers */
void rastrigin_func (double *, double *, int , double *, int); /* Rastrigin's  */
void schwefel_func (double *, double *, int , double *, int); /* Schwefel's */
void bi_rastrigin_func (double *, double *, int , double *, int); /* Lunacek Bi_rastrigin */
void cf02 (double *, double *, int , double *,  int); /* Composition Function 2 */

void shiftfunc (double*,double*,int,double*);
void rotatefunc (double*,double*,int, double*);
void asyfunc (double *, double *x, int, double);
void oszfunc (double *, double *, int);
void cf_cal(double *, double *, int, double *,double *,double *,double *,int);

void test_func(double *x, double *f, double *m_d, double *OShift, int nx, int mx,int func_num, int cf_num)
{
	int i;
	if (ini_flag==1)
	{
		if ((n_flag!=nx)||(func_flag!=func_num))
		{
			ini_flag=0;
		}
	}

	if (ini_flag==0)
	{
//		free(M);
//		free(OShift);
		ldm_free(y);
		ldm_free(z);
		ldm_free(x_bound);
//    double MEM[128];
//    y = MEM;
//    z = y + nx;
//    x_bound = z + nx; 
		y=(double *)ldm_malloc(sizeof(double)  *  nx);
		z=(double *)ldm_malloc(sizeof(double)  *  nx);
		x_bound=(double *)ldm_malloc(sizeof(double)  *  nx);
		for (i=0; i<nx; i++)
			x_bound[i]=100.0;

		if (!(nx==2||nx==5||nx==10||nx==20||nx==30||nx==40||nx==50||nx==60||nx==70||nx==80||nx==90||nx==100))
		{
			printf("\nError: Test functions are only defined for D=2,5,10,20,30,40,50,60,70,80,90,100.\n");
		}

		n_flag=nx;
		func_flag=func_num;
		ini_flag=1;
	}
  int my_id = athread_get_id(-1);
//  if (my_id == 1) {
//    for (i=0; i<cf_num*nx; i++) {
//      printf("OShift[%d] = %lf\n", i, OShift[i]);
//    }
//  }
	for (i = 0; i < mx; i++) {
      switch(func_num)
      {
        case 1:
			//sphere_func(&x[i*nx],&f[i],nx,OShift,M,0);
			    sphere_func(&x[i*nx],&f[i],nx,OShift,0);
          f[i]+=-1400.0;
          break;
		    case 5:
			    dif_powers_func(&x[i*nx],&f[i],nx,OShift,0);
			    f[i]+=-1000.0;
			    break;
		    case 11:	
			    rastrigin_func(&x[i*nx],&f[i],nx,OShift,0);
			    f[i]+=-400.0;
			    break;
		    case 14:	
			    schwefel_func(&x[i*nx],&f[i],nx,OShift,0);
			    f[i]+=-100.0;
			    break;
		    case 17:	
			    bi_rastrigin_func(&x[i*nx],&f[i],nx,OShift,0);
			    f[i]+=300.0;
			    break;
		    case 22:	
			    cf02(&x[i*nx],&f[i],nx,OShift,0);
			    f[i]+=800.0;
			    break;
		    default:
			    printf("\nError: There are only 28 test functions in this test suite!\n");
			    f[i] = 0.0;
			    break;
      }
  }
}

void rastrigin_func (double *x, double *f, int nx, double *Os,int r_flag) /* Rastrigin's  */
{
    int i;
	double alpha=10.0,beta=0.2;
	shiftfunc(x, y, nx, Os);
	for (i=0; i<nx; i++)//shrink to the orginal search range
    {
        y[i]=y[i]*5.12/100;
    }

	//if (r_flag==1)
	//rotatefunc(y, z, nx, Mr);
	//else
  for (i=0; i<nx; i++)
		z[i]=y[i];

    oszfunc (z, y, nx);
    asyfunc (y, z, nx, beta);

	//if (r_flag==1)
	//rotatefunc(z, y, nx, &Mr[nx*nx]);
	//else
  for (i=0; i<nx; i++)
		y[i]=z[i];

	for (i=0; i<nx; i++)
	{
		y[i]*=pow(alpha,1.0*i/(nx-1)/2);
	}

	//if (r_flag==1)
	//rotatefunc(y, z, nx, Mr);
	//else
  for (i=0; i<nx; i++)
		z[i]=y[i];

    f[0] = 0.0;
    for (i=0; i<nx; i++)
    {
        f[0] += (z[i]*z[i] - 10.0*cos(2.0*PI*z[i]) + 10.0);
    }
}

void sphere_func (double *x, double *f, int nx, double *Os, int r_flag) /* Sphere */
{
	int i;
  int my_id = athread_get_id(-1);
	shiftfunc(x, y, nx, Os);
	//if (r_flag==1)
	//rotatefunc(y, z, nx, m_d);
  for (i=0; i<nx; i++) {
		z[i]=y[i];
  //  if(my_id == 1) {
  //    printf("z[%d] = %lf\n", i, z[i]);
  //  }
  }
	f[0] = 0.0;
    for (i=0; i<nx; i++)
    {
        f[0] += z[i]*z[i];
     //   if(my_id == 1) {
     //     printf("f = %lf\n", f[0]);
     //   }
    }
}

void dif_powers_func (double *x, double *f, int nx, double *Os,int r_flag) /* Different Powers */
{
	int i;
	shiftfunc(x, y, nx, Os);
	//if (r_flag==1)
	//rotatefunc(y, z, nx, Mr);
	//else
  for (i=0; i<nx; i++)
		z[i]=y[i];
	f[0] = 0.0;
    for (i=0; i<nx; i++)
    {
        f[0] += pow(fabs(z[i]),2+4*i/(nx-1));
    }
	f[0]=pow(f[0],0.5);
}

void schwefel_func (double *x, double *f, int nx, double *Os, int r_flag) /* Schwefel's  */
{
    int i;
	double tmp;
	shiftfunc(x, y, nx, Os);
	for (i=0; i<nx; i++)//shrink to the orginal search range
    {
        y[i]*=1000/100;
    }
	//if (r_flag==1)
	//rotatefunc(y, z, nx, Mr);
	//else
  for (i=0; i<nx; i++)
		z[i]=y[i];

	for (i=0; i<nx; i++)
		y[i] = z[i]*pow(10.0,1.0*i/(nx-1)/2.0);

	for (i=0; i<nx; i++)
		z[i] = y[i]+4.209687462275036e+002;
	
    f[0]=0;
    for (i=0; i<nx; i++)
	{
		if (z[i]>500)
		{
			f[0]-=(500.0-fmod(z[i],500))*sin(pow(500.0-fmod(z[i],500),0.5));
			tmp=(z[i]-500.0)/100;
			f[0]+= tmp*tmp/nx;
		}
		else if (z[i]<-500)
		{
			f[0]-=(-500.0+fmod(fabs(z[i]),500))*sin(pow(500.0-fmod(fabs(z[i]),500),0.5));
			tmp=(z[i]+500.0)/100;
			f[0]+= tmp*tmp/nx;
		}
		else
			f[0]-=z[i]*sin(pow(fabs(z[i]),0.5));
    }
    f[0]=4.189828872724338e+002*nx+f[0];
}

void bi_rastrigin_func (double *x, double *f, int nx, double *Os, int r_flag) /* Lunacek Bi_rastrigin Function */
{
    int i;
	double mu0=2.5,d=1.0,s,mu1,tmp,tmp1,tmp2;
//  double MEM[128];
	double *tmpx;
//  tmpx = MEM;
	tmpx=(double *)ldm_malloc(sizeof(double)  *  nx);
	s=1.0-1.0/(2.0*pow(nx+20.0,0.5)-8.2);
	mu1=-pow((mu0*mu0-d)/s,0.5);

	shiftfunc(x, y, nx, Os);
	for (i=0; i<nx; i++)//shrink to the orginal search range
    {
        y[i]*=10.0/100.0;
    }

	for (i = 0; i < nx; i++)
    {
		tmpx[i]=2*y[i];
        if (Os[i] < 0.)
            tmpx[i] *= -1.;
    }

	for (i=0; i<nx; i++)
	{
		z[i]=tmpx[i];
		tmpx[i] += mu0;
	}
	//if (r_flag==1)
	//rotatefunc(z, y, nx, Mr);
	//else
  for (i=0; i<nx; i++)
		y[i]=z[i];

	for (i=0; i<nx; i++)
		y[i] *=pow(100.0,1.0*i/(nx-1)/2.0);
	//if (r_flag==1)
	//rotatefunc(y, z, nx, &Mr[nx*nx]);
	//else
  for (i=0; i<nx; i++)
		z[i]=y[i];

    tmp1=0.0;tmp2=0.0;
    for (i=0; i<nx; i++)
	{
		tmp = tmpx[i]-mu0;
		tmp1 += tmp*tmp;
		tmp = tmpx[i]-mu1;
		tmp2 += tmp*tmp;
    }
	tmp2 *= s;
	tmp2 += d*nx;
	tmp=0;
	for (i=0; i<nx; i++)
	{
		tmp+=cos(2.0*PI*z[i]);
    }
	
	if(tmp1<tmp2)
		f[0] = tmp1;
	else
		f[0] = tmp2;
	f[0] += 10.0*(nx-tmp);
	ldm_free(tmpx);
}

void cf02 (double *x, double *f, int nx, double *Os,int r_flag) /* Composition Function 2 */
{
	int i,cf_num=3;
	double fit[3];
	double delta[3] = {20,20,20};
	double bias[3] = {0, 100, 200};
	for(i=0;i<cf_num;i++)
	{
		schwefel_func(x,&fit[i],nx,&Os[i*nx],r_flag);
	}
	cf_cal(x, f, nx, Os, delta,bias,fit,cf_num);
}

void shiftfunc (double *x, double *xshift, int nx,double *Os)
{
	int i;
    for (i=0; i<nx; i++)
    {
        xshift[i]=x[i]-Os[i];
    }
}

void rotatefunc (double *x, double *xrot, int nx,double *m_d)
{
	int i,j;
    for (i=0; i<nx; i++)
    {
        xrot[i]=0;
			for (j=0; j<nx; j++)
			{
				xrot[i]=xrot[i]+x[j]*m_d[i*nx+j];
			}
    }
}

void asyfunc (double *x, double *xasy, int nx, double beta)
{
	int i;
    for (i=0; i<nx; i++)
    {
		if (x[i]>0)
        xasy[i]=pow(x[i],1.0+beta*i/(nx-1)*pow(x[i],0.5));
    }
}

void oszfunc (double *x, double *xosz, int nx)
{
	int i,sx;
	double c1,c2,xx;
    for (i=0; i<nx; i++)
    {
		if (i==0||i==nx-1)
        {
			if (x[i]!=0)
				xx=log(fabs(x[i]));
			if (x[i]>0)
			{	
				c1=10;
				c2=7.9;
			}
			else
			{
				c1=5.5;
				c2=3.1;
			}	
			if (x[i]>0)
				sx=1;
			else if (x[i]==0)
				sx=0;
			else
				sx=-1;
			xosz[i]=sx*exp(xx+0.049*(sin(c1*xx)+sin(c2*xx)));
		}
		else
			xosz[i]=x[i];
    }
}

void cf_cal(double *x, double *f, int nx, double *Os,double * delta,double * bias,double * fit, int cf_num)
{
	int i,j;
	double *w;
	double w_max=0,w_sum=0;
//  double MEM[MAX_CF];
//  w = MEM;
	w=(double *)ldm_malloc(cf_num * sizeof(double));
	for (i=0; i<cf_num; i++)
	{
		fit[i]+=bias[i];
		w[i]=0;
		for (j=0; j<nx; j++)
		{
			w[i]+=pow(x[j]-Os[i*nx+j],2.0);
		}
		if (w[i]!=0)
			w[i]=pow(1.0/w[i],0.5)*exp(-w[i]/2.0/nx/pow(delta[i],2.0));
		else
			w[i]=INF;
		if (w[i]>w_max)
			w_max=w[i];
	}

	for (i=0; i<cf_num; i++)
	{
		w_sum=w_sum+w[i];
	}
	if(w_max==0)
	{
		for (i=0; i<cf_num; i++)
			w[i]=1;
		w_sum=cf_num;
	}
	f[0] = 0.0;
    for (i=0; i<cf_num; i++)
    {
		f[0]=f[0]+w[i]/w_sum*fit[i];
    }
	ldm_free(w);
}


void compute_genetic(genetic_array *ga) {
  int my_id, i;
  volatile int get_reply, put_reply;
  volatile genetic_array slave_ga;
  my_id = athread_get_id(-1);
  // for(i = 0; i < 640*MAX_CF; i++){
	  // if (my_id == 1)
	  // printf("%d\n",ga->shift_data);
  // }
  get_reply = 0;  
  athread_get(PE_MODE, ga, &slave_ga, sizeof(genetic_array), &get_reply, 0, 0, 0);
  while(get_reply!=1);
  //printf(">\n");
  volatile int n = slave_ga.n;
  volatile int m = slave_ga.m;
  volatile int func_num = slave_ga.func_num;
  double *m_d = (double *)slave_ga.m_d;
  double *gen_x = (double *)slave_ga.x;
  double *gen_OShift = (double *)slave_ga.OShift;
  double *gen_f = (double *)slave_ga.f;
  int cf_num = 10;
  int size = n*m + n*cf_num + m;
 
  double MEM[size];
  double *x = MEM;
  double *f = x + m*n;
  //M = f + m;
  OShift = f + m;
  get_reply = 0;
  athread_get(PE_MODE, gen_x+my_id*n, x, sizeof(double)*n*m, &get_reply, 0, 0, 0);
  athread_get(PE_MODE, gen_OShift, OShift, sizeof(double)*n*cf_num, &get_reply, 0, 0, 0);
  //athread_get(PE_MODE, m_d, M, sizeof(double)*cf_num*n*n, &get_reply, 0, 0, 0);
  while(get_reply!=2);

  int  j;
 //   for (i = 0; i < m; i++)
 //   {
 //   	for (j = 0; j < n; j++)
 //   	{
 //   		x[i*n+j]=0.0;
 //   	}
 //   }
 // if (my_id == 2) {
 //     for (i = 0; i < m; i++)
 //     {
 //     	for (j = 0; j < n; j++)
 //     	{
 //     		printf("slave_x[%d] = %lf, ",j, x[i*n+j]);
 //     	}
 //     }
 //     printf("\n");
 //     for (i = 0; i < m; i++)
 //     {
 //     	for (j = 0; j < n*cf_num; j++)
 //     	{
 //     		printf("OShift[%d] = %lf, ",j, x[i*n+j]);
 //     	}
 //     }
 //     printf("\n");
 //}
 // }
  //printf(">>>>");
  put_reply = 0;
  test_func(x, f, m_d, OShift, n, m, func_num, cf_num);
  athread_put(PE_MODE, f, gen_f+my_id*m, sizeof(double)*m, &put_reply, 0, 0); 

  //printf("f%d(x[1]) = %Lf,", func_num,f);
}
