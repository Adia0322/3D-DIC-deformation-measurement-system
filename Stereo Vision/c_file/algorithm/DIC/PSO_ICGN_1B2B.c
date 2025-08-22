/*  PSO_1B2B 2022.03.06 23:22 */ 
/* scan must be odd number, ex:101 */ 
/* Coefficient criterion: ZNCC */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define square(x) ((x)*(x))
#define mean(x) ((x)/(double)(Size*Size))

#define img_row 480 
#define img_col 640 

#define Size 31 
#define SizeHalf (Size-1)/2 
#define scan 31
// PSO 
#define Population 100
#define Dimension 2
#define Iteration 20
#define Iter_reciprocal (1.0/Iteration)
#define ArraySize_Pini 5
#define FixedPointRange scan/2
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
double Cost_function(int Pi_y, int Pi_x, int Object_point[], int img_aft[][img_col],\
                     int img_aft_sub[][Size], int img_bef_sub[][Size], double Mean_bef[]);
double GRandom(void);
void init_random_seed();
void print_array(int array[][Size]);

/* search (Main)  */
__declspec(dllexport)
void SCAN(int img_aft[][img_col], int img_aft_sub[][Size], int img_bef_sub[][Size],\
          double Mean_bef[], int Object_point[], int Displacement[], double CoefValue[])
{
	int i, j, k, Pi_y_ini, Pi_x_ini, Pi_y, Pi_x, Count_y=0, Count_x=0;
	int max_index = 0; 
	double Pbest[Population][Dimension], Gbest[Dimension];   /* Gbest[0] = y, Gbest[1]  = x */
	double upper_bounds[2]={Boundary_Length, Boundary_Length}, lower_bounds[2]={-Boundary_Length, -Boundary_Length}; 
	double Pi[Population][Dimension], Vi[Population][Dimension];
	double Cost_initial, Cost, max_value_Gbest=-1e+9, max_value_Pbest[Population];
	double Gbest_y, Gbest_x;
	
	// initialize random var (only do one time)
	init_random_seed();
	
	for (i=0;i<Population;i++)
	{
		//(ArraySize_Pini)*(ArraySize_Pini)
		if (i<(ArraySize_Pini*ArraySize_Pini))
		{
			Vi[i][0] = Vini*(GRandom()*2-1);
			Vi[i][1] = Vini*(GRandom()*2-1);
			if (Count_y == ArraySize_Pini)
			{
				Count_y = 0;
				Count_x += 1;
			}
			Pi[i][0] = -Array_Start + Count_y*ArrayInterval;
			Pi[i][1] = -Array_Start + Count_x*ArrayInterval;
			
			Pbest[i][0] = Pi[i][0];
			Pbest[i][1] = Pi[i][1];
			
			/* Calculate SSD (sum of squared differences) */
			Pi_y_ini = (int)Pi[i][0];
			Pi_x_ini = (int)Pi[i][1];
			Cost_initial = Cost_function(Pi_y_ini, Pi_x_ini, Object_point, img_aft,\
			                             img_aft_sub, img_bef_sub, Mean_bef);
			
			/* Individual best value */
			max_value_Pbest[i] = Cost_initial;
			
			/* Global best value */
			if (Cost_initial>max_value_Gbest)
			{
				max_value_Gbest = Cost_initial;
				max_index = i;
				
				Gbest[0] = Pi[max_index][0];
				Gbest[1] = Pi[max_index][1];
			}
			Count_y += 1;
		}
		
		else
		{
			for (j=0;j<Dimension;j++)
			{
				Vi[i][j] = Vini*(GRandom()*2-1);
				Pi[i][j] = lower_bounds[j] + 0.5*Boundary_Length +\
				           0.5*GRandom()*(upper_bounds[j]-lower_bounds[j]);
				
				Pbest[i][j] = Pi[i][j];
			}
			/* Calculate SSD (sum of squared differences) */
			Pi_y_ini = (int)Pi[i][0];
			Pi_x_ini = (int)Pi[i][1];
			Cost_initial = Cost_function(Pi_y_ini, Pi_x_ini, Object_point, img_aft,\
			                             img_aft_sub, img_bef_sub, Mean_bef);
			
			/* Individual best value */
			max_value_Pbest[i] = Cost_initial;
			
			/* Global best value */
			if (Cost_initial>max_value_Gbest)
			{
				max_value_Gbest = Cost_initial; //max_value_Gbest
				max_index = i; /*  (u,v) = (Pi[max_index,0] , Pi[max_index,1]) */
				
				Gbest[0] = Pi[max_index][0];
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
					Vi[i][j] = Vmax;
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
			/* Calculate ZNCC (sum of squared differences) */
			Pi_y = (int)Pi[i][0]; /* array only accept integer as Argument */
			Pi_x = (int)Pi[i][1];
			Cost = Cost_function(Pi_y, Pi_x, Object_point, img_aft,\
			                     img_aft_sub, img_bef_sub, Mean_bef);
			
			/* Individual best value */
			if (Cost>max_value_Pbest[i])
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
	Gbest_y = Gbest[0];
	Gbest_x = Gbest[1];
	Displacement[0] = (int)Gbest_y; /* vertical (down:+) */
	Displacement[1] = (int)Gbest_x; /* horizontal (right:+) */
	CoefValue[0] = max_index; /* The point index of the global maximum value */
	CoefValue[1] = max_value_Gbest; /* The global maximum value */
}


/*============================ Functions ==============================*/
/*ZNCC(zero-normalized cross-correlation)(-1 ~ +1) */
double Cost_function(int Pi_y, int Pi_x, int Object_point[], int img_aft[][img_col],\
                     int img_aft_sub[][Size], int img_bef_sub[][Size], double Mean_bef[])   
{
	int i, j, Aft_sub_sum=0;
	int row, col;
	double Mean_aft, Sum_Numerator=0.0, Sum_Denominator_bef=0.0, Sum_Denominator_aft=0.0, coef=0.0;
	/* Construct img_aft_sub */
	for (i=0;i<Size;i++) 
	{
		for (j=0;j<Size;j++)
		{
			row = i - SizeHalf + Pi_y + Object_point[0];
			col = j - SizeHalf + Pi_x + Object_point[1];
			if(row>=img_row || row<0 || col>=img_col || col<0){
				printf("\n[ERROR] row>img_row || row<0 || col>img_col || col<img_col\n");
				exit(1);
			}
			img_aft_sub[i][j] = img_aft[row][col];
			if (img_aft_sub[i][j]>255 || img_aft_sub[i][j]<0){
				printf("\n[ERROR] img_aft_sub[i][j]>255 || img_aft_sub[i][j]<00");
				exit(1);
			}
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
	if(Sum_Denominator_bef==0 || Sum_Denominator_aft==0){
		return 0.0;
	}
	coef = Sum_Numerator/(sqrt(Sum_Denominator_bef*Sum_Denominator_aft));
	return coef;
}

/* Generate random number */
// double GRandom(void)
// {
// 	double i = fmod(rand(),1000.0)/1000.0;
// 	return i;
// }


double GRandom(void)
{
    return rand() / (double)RAND_MAX;
}


void init_random_seed() {
    static int seeded = 0;
    if (!seeded) {
        time_t t = time(NULL);
        srand((unsigned) t);
        seeded = 1;
    }
}

void print_array(int array[][Size]){
	for (int i = 0; i < Size; i++)
	{	
		for (int j = 0; j < Size; j++)
		{
			printf("%d ",array[i][j]);
		}
		printf("\n");
	}
	return;
}