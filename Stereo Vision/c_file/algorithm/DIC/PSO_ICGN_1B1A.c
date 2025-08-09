/*  PSO_ICGN_1B1A 2022.03.06 23:22 */ 
/* scan must be odd number, ex:101 */ 
/* Coefficient criterion: ZNCC */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
/* �ƾǦ� */
#define square(x) ((x)*(x))
#define mean(x) ((x)/(Size*Size))
/* �Ϲ��]�w */
#define img_row 480 
#define img_column 640 
/* DIC �ѼƳ]�w */ 
#define Size 31  /* ��Τl���X��� */ 
#define SizeHalf (Size-1)/2 
#define scan 31        /* ��α��y�ϰ���� */ 
/* PSO �ѼƳ]�w */ 
#define Population 20 
#define Dimension 2
#define Iteration 4
#define Iter_reciprocal (1.0/Iteration)
#define ArraySize_Pini 3      /* �]�wn*n��}�I����l�T�w��m(n���_��)�A�Ϫ�l�j�M��[���áA���`�NPopulation������!! */ 
#define FixedPointRange scan/2   /* �]�wn*n��}�I���d�� (��}���) */ 
#define ArrayInterval (FixedPointRange)/(ArraySize_Pini-1) 
#define Array_Start ArrayInterval*(ArraySize_Pini-1)/2
#define Boundary_Length 0.5*(scan-1)
#define Vmax 0.5*Boundary_Length
#define Vini 0.2*Vmax
#define W_upper 0.9 
#define W_lower 0.4
#define Decrease_factor 1 
#define Increase_factor 1.05
#define Cognition_factor 1.0 
#define Social_factor 1.0 

/* functions */
double Cost_function(int Pi_u, int Pi_v, int Object_point[], int img_aft[][img_column],\
                     int img_aft_sub[][Size], int img_bef_sub[][Size], double Mean_bef[]);
double GRandom(void);

/* construct C matrix (Main)  */ 
void SCAN(int img_aft[][img_column], int img_aft_sub[][Size], int img_bef_sub[][Size],\
          double Mean_bef[], int Object_point[], int Displacement[], double CoefValue[])
{
	int i, j, k, m, x, y, Pi_u_ini, Pi_v_ini, Pi_u, Pi_v, Count_u=0, Count_v=0; /*  */ 
	int max_index = 0; 
	double Pbest[Population][Dimension], Gbest[Dimension];   /* Gbest[0] = x, Gbest[1]  = y */
	double upper_bounds[2]={Boundary_Length, Boundary_Length}, lower_bounds[2]={-Boundary_Length, -Boundary_Length}; 
	double Pi[Population][Dimension], Vi[Population][Dimension];
	double Cost_initial, Cost, max_value_Gbest=-1e+9, max_value_Pbest[Population]; /* �]�wmax_value_Gbest�ɼƭȺɶq�p */ 
	double Gbest_u, Gbest_v;
	
	// initialize random var (only do one time)
	static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
	
	for (i=0;i<Population;i++) /* i: �I�s�� */ 
	{
		/* �]�w�T�w�t�׻P��m���I �ƶq:(ArraySize_Pini)*(ArraySize_Pini) */ 
		if (i<(ArraySize_Pini*ArraySize_Pini))  /* �T�{�ɤl�s���p����T�w���ɤl�ƶq */ 
		{
			Vi[i][0] = Vini*(GRandom()*2-1); /* �гy -1~1 ���p��  note: Vini*(-1.0 ~ 1.0)  */
			Vi[i][1] = Vini*(GRandom()*2-1);
			if (Count_u == ArraySize_Pini)
			{
				Count_u = 0;
				Count_v += 1;
			}
			Pi[i][0] = -Array_Start + Count_u*ArrayInterval;
			Pi[i][1] = -Array_Start + Count_v*ArrayInterval;
			
			Pbest[i][0] = Pi[i][0]; /* �N��i���I��u�Bv��(j=0��u, j=1��v)�����A�b��l��Pi[i][j]�Y���ӤH�g��̨Φ�mPbest[i][j] */
			Pbest[i][1] = Pi[i][1];
			
			/* Calculate SSD (sum of squared differences) */
			Pi_u_ini = (int)Pi[i][0]; /* �j���૬�u�h���p�ơA���|�|�ˤ��J */
			Pi_v_ini = (int)Pi[i][1];
			Cost_initial = Cost_function(Pi_u_ini, Pi_v_ini, Object_point,\
			                             img_aft, img_aft_sub, img_bef_sub, Mean_bef);
			
			/* Individual best value */
			max_value_Pbest[i] = Cost_initial; /* �@�}�l�ӤH�̨θg��u��1�� */
			
			/* Global best value */
			if (Cost_initial>max_value_Gbest)
			{
				max_value_Gbest = Cost_initial;
				max_index = i;
				
				Gbest[0] = Pi[max_index][0];
				Gbest[1] = Pi[max_index][1];
			}
			
			Count_u += 1;
		}
		
		else
		{
			/* �]�w�H���t�׻P��m���I */
			for (j=0;j<Dimension;j++)
			{
				Vi[i][j] = Vini*(GRandom()*2-1); /* Vini*(-1.0 ~ 1.0) */
				Pi[i][j] = lower_bounds[j] + 0.5*Boundary_Length +\
				           0.5*GRandom()*(upper_bounds[j]-lower_bounds[j]); /* �`�N!! Vi�BPi���B�I�ƫ��A */
				
				Pbest[i][j] = Pi[i][j]; /* �N��i���I��u�Bv��(j=0��u, j=1��v)�����A�b��l��Pi[i][j]�Y���ӤH�g��̨Φ�mPbest[i][j] */
			}
			/* Calculate SSD (sum of squared differences) */
			Pi_u_ini = (int)Pi[i][0]; /* �j���૬�u�h���p�ơA���|�|�ˤ��J */ 
			Pi_v_ini = (int)Pi[i][1];
			Cost_initial = Cost_function(Pi_u_ini, Pi_v_ini, Object_point,\
			                             img_aft, img_aft_sub, img_bef_sub, Mean_bef);  /* �`�N!! �o�̤���Dimension���ܤơA�Y�n�󴫺��׽Ъ`�N!!!*/ 
			
			/* Individual best value */
			max_value_Pbest[i] = Cost_initial; /* �@�}�l�ӤH�̨θg��u��1�� */ 
			
			/* Global best value */
			if (Cost_initial>max_value_Gbest)
			{
				max_value_Gbest = Cost_initial;  /* max_value_Gbest���s��̨Φ�m���ȡA�N�b����C�����N����s */
				max_index = i; /* max_index���s��̨Φ�m���I�s���A�s��̨Φ�m: (u,v) = (Pi[max_index,0] , Pi[max_index,1]) */
				
				Gbest[0] = Pi[max_index][0]; /* Gbest[0]���ܸs��̨Φ�mx�ȡAGbest[1]�h��y�ȡC   �`�N!! �o�̤���Dimension���ܤơA�Y�n�󴫺��׽Ъ`�N!!!*/
				Gbest[1] = Pi[max_index][1];
			}
		}
	}
	
	/* Start iteration */ 
	
	for (k=0;k<Iteration;k++)
	{
		for (i=0;i<Population;i++)
		{
			for (j=0;j<Dimension;j++)
			{
				Vi[i][j] = (W_upper-(k+1)*(W_upper-W_lower)*Iter_reciprocal)*Vi[i][j] +\
				            Cognition_factor*GRandom()*(Pbest[i][j]-Pi[i][j]) +\
				            Social_factor*GRandom()*(Gbest[j]-Pi[i][j]);
				if (Vi[i][j]>Vmax)
				{
					Vi[i][j] = Vmax; /* ����t�פW�� */ 
				}
				if (Vi[i][j]<-Vmax)
				{
					Vi[i][j] = -Vmax;
				}
				
				Pi[i][j] += Vi[i][j];  
				
				if (Pi[i][j]>upper_bounds[j]) /* restrict boundary */ 
				{
					Pi[i][j] = upper_bounds[j];
				}
				
				if (Pi[i][j]<lower_bounds[j])
				{
					Pi[i][j] = lower_bounds[j];
				}
			}
			
			/* Calculate SSD (sum of squared differences) */
			Pi_u = (int)Pi[i][0]; /* array only accept integer as Argument */
			Pi_v = (int)Pi[i][1];
			Cost = Cost_function(Pi_u, Pi_v, Object_point, img_aft,\
			                     img_aft_sub, img_bef_sub, Mean_bef);
			
			/* Individual best value */
			if (Cost>max_value_Pbest[i]) /* �C�����NPbest��s���ӤH�g��̨Φ�m */
			{
				max_value_Pbest[i] = Cost;
				Pbest[i][0]=Pi[i][0];
				Pbest[i][1]=Pi[i][1];
				
				/* Global best value */
				if (Cost>max_value_Gbest)
				{
					max_value_Gbest = Cost;
					max_index = i;
					
					Gbest[0] = Pi[max_index][0];
					Gbest[1] = Pi[max_index][1];
				}
			}
			/* sensor */
			/*sensor[i][0] = (int)Pi[i][0];*/
			/*sensor[i][1] = (int)Pi[i][1];*/
		}
		
	}
	/*sensor_coef[0] = Cost_function(0, 0, Object_point, img_aft, img_aft_sub, img_bef_sub);*/
	/* Output Result */
	Gbest_u = Gbest[0];
	Gbest_v = Gbest[1];
	Displacement[0] = (int)Gbest_u; /* vertical (down:+) */
	Displacement[1] = (int)Gbest_v; /* horizontal (right:+) */
	CoefValue[0] = max_index; /* The point index of the global maximum value */
	CoefValue[1] = max_value_Gbest; /* The global maximum value */
}


/*============================ Functions ==============================*/
/* �̾� ZNCC(zero-normalized cross-correlation)�����Y�Ʒǫh �p������Y�� (-1 ~ +1) */
double Cost_function(int Pi_u, int Pi_v, int Object_point[], int img_aft[][img_column],\
                     int img_aft_sub[][Size], int img_bef_sub[][Size], double Mean_bef[])   
{
	int i, j, Aft_sub_sum=0;
	double Mean_aft, Sum_Numerator=0.0, Sum_Denominator_bef=0.0, Sum_Denominator_aft=0.0, coef=0.0;
	/* Construct img_aft_sub */
	for (i=0;i<Size;i++) 
	{
		for (j=0;j<Size;j++)
		{
			img_aft_sub[i][j] =\
			 img_aft[i - SizeHalf + Pi_u + Object_point[0]][j - SizeHalf + Pi_v + Object_point[1]]; /*   */ 
		}
	}
	/* Mean of img_aft_sub */
	for (i=0;i<Size;i++)
	{
		for (j=0;j<Size;j++)
		{
			Aft_sub_sum+=img_aft_sub[i][j];
		}
	}
	Mean_aft=mean(Aft_sub_sum); /* mean function is defined by macro (#define)   */ 
	/* Substract its mean, comopute its sqrt and sum */
	for (i=0;i<Size;i++)
	{
		for (j=0;j<Size;j++)
		{
			Sum_Numerator+=(img_bef_sub[i][j] - Mean_bef[0])*(img_aft_sub[i][j] - Mean_aft);
			Sum_Denominator_bef+=square(img_bef_sub[i][j] - Mean_bef[0]);
			Sum_Denominator_aft+=square(img_aft_sub[i][j] - Mean_aft);
		}
	}
	coef = Sum_Numerator/(sqrt(Sum_Denominator_bef*Sum_Denominator_aft));
	return coef;
}

/* Generate random number */
double GRandom(void)
{
	double i = fmod(rand(),1000.0)/1000.0;
	return i;
}

/*
double GRandom(void)
{
    return rand() / (double)RAND_MAX;
}
*/