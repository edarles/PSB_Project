#include <PeriodicHeightField.h>
#include <PeriodicHeightField.cuh>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
/*****************************************************************************************************/
/*****************************************************************************************************/
namespace Utils {

/*****************************************************************************************************/
/*****************************************************************************************************/
Periodic_HeightField::Periodic_HeightField():HeightField()
{
}
/*****************************************************************************************************/
Periodic_HeightField::Periodic_HeightField(uint nbFunc, double AMin, double AMax, double kMin, double kMax, double thetaMin, double thetaMax,
				          double phiMin, double phiMax, double wMin, double wMax, Vector3 Min, Vector3 Max, double d_x, double d_z)
		     :HeightField(Min,Max,d_x,d_z)
{
	create(nbFunc,AMin,AMax,kMin,kMax,thetaMin,thetaMax,phiMin,phiMax,wMin,wMax);
}
/*****************************************************************************************************/
Periodic_HeightField::Periodic_HeightField(const Periodic_HeightField& H):HeightField(H)
{
	this->nbFunc = H.nbFunc;
	this->AMin = H.AMin; this->AMax = H.AMax;
	this->kMin = H.kMin; this->kMax = H.kMax;
	this->thetaMin = H.thetaMin; this->thetaMax = H.thetaMax;
	this->phiMin = H.phiMin; this->phiMax = H.phiMax;
	this->wMin = H.wMin; this->wMax = H.wMax;
}
/*****************************************************************************************************/
Periodic_HeightField::~Periodic_HeightField()
{
	nbFunc = 0;
	freeArray(m_A);
	freeArray(m_k);
	freeArray(m_theta);
	freeArray(m_phi);
	freeArray(m_w);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void  Periodic_HeightField::create(uint nbFunc, double AMin, double AMax, double kMin, double kMax, double thetaMin, double thetaMax, 
				   double phiMin, double phiMax, double wMin, double wMax)
{
	this->nbFunc = nbFunc;
	this->AMin = AMin; this->AMax = AMax;
	this->kMin = kMin; this->kMax = kMax;
	this->thetaMin = thetaMin; this->thetaMax = thetaMax;
	this->phiMin = phiMin; this->phiMax = phiMax;
	this->wMin = wMin; this->wMax = wMax;
	initialize();
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Periodic_HeightField::initialize()
{
	if(m_A!=NULL) freeArray(m_A);
	if(m_k!=NULL) freeArray(m_k);
	if(m_theta!=NULL) freeArray(m_theta);
	if(m_phi!=NULL) freeArray(m_phi);
	if(m_w!=NULL) freeArray(m_w);

        double *A, *k, *theta, *phi, *w;
	A = new double[nbFunc];
	k = new double[nbFunc];
	theta = new double[nbFunc];
	phi = new double[nbFunc];
	w = new double[nbFunc];

	srand(time(NULL));
	for(uint i=0;i<nbFunc;i++){
		A[i] = rand()/((double)RAND_MAX )*(AMax-AMin) + AMin;
		k[i] = rand()/((double)RAND_MAX )*(kMax-kMin) + kMin;
		theta[i] = rand()/((double)RAND_MAX )*(thetaMax-thetaMin) + thetaMin;
		phi[i] = rand()/((double)RAND_MAX )*(phiMax-phiMin) + phiMin;
		w[i] = rand()/((double)RAND_MAX )*(wMax-wMin) + wMin;
		printf("A:%f k:%f theta:%f phi:%f w:%f\n",A[i],k[i],theta[i],phi[i],w[i]);
	}
	allocateArray((void**)&m_A, nbFunc*sizeof(double));
	allocateArray((void**)&m_k, nbFunc*sizeof(double));
	allocateArray((void**)&m_theta, nbFunc*sizeof(double));
	allocateArray((void**)&m_phi, nbFunc*sizeof(double));
	allocateArray((void**)&m_w, nbFunc*sizeof(double));

	copyArrayToDevice(m_A, A, 0, nbFunc*sizeof(double));
	copyArrayToDevice(m_k, k, 0, nbFunc*sizeof(double));
	copyArrayToDevice(m_theta, theta, 0, nbFunc*sizeof(double));
	copyArrayToDevice(m_phi, phi, 0, nbFunc*sizeof(double));
	copyArrayToDevice(m_w, w, 0, nbFunc*sizeof(double));

	delete[] A;
	delete[] k;
	delete[] theta;
	delete[] phi;
	delete[] w;

}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Periodic_HeightField::calculateHeight(double* m_pos, uint nbPos)
{
	PeriodicHeightField_calculateHeight_CUDA(m_pos,m_A,m_k,m_theta,m_phi,nbFunc,nbPos);
	calculateHeight_Normales_Gradient(m_pos,m_N,nbPos);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Periodic_HeightField::calculateHeight_Normales(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1, uint nbPos)
{
	PeriodicHeightField_calculateHeight_Normales_CUDA(m_pos,nx0,nx1,nz0,nz1,m_A,m_k,m_theta,m_phi,nbFunc,nbPos);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Periodic_HeightField::calculateHeight_Normales_Gradient(double* m_pos, double *m_N, uint nbPos)
{
	PeriodicHeightField_calculateHeight_Normales_Gradient_CUDA(m_pos, m_N,m_A,m_k,m_theta,m_phi,nbFunc,nbPos);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Periodic_HeightField::saveSpectrum(const char* filename)
{
	FILE *f = fopen(filename,"w");
	if(f!=NULL && nbFunc>0){
		fprintf(f,"%d\n",nbFunc);

		double *A, *k, *theta, *phi, *w;
		A = new double[nbFunc];
		k = new double[nbFunc];
		theta = new double[nbFunc];
		phi = new double[nbFunc];
		w = new double[nbFunc];

		copyArrayFromDevice(A, m_A, 0, nbFunc*sizeof(double));
		copyArrayFromDevice(k, m_k, 0, nbFunc*sizeof(double));
		copyArrayFromDevice(theta, m_theta, 0, nbFunc*sizeof(double));
		copyArrayFromDevice(phi, m_phi, 0, nbFunc*sizeof(double));
		copyArrayFromDevice(w, m_w, 0, nbFunc*sizeof(double));

		for(uint i=0;i<nbFunc;i++){
			fprintf(f,"A:%lf k:%lf theta:%lf phi:%lf w:%lf\n",A[i],k[i],theta[i],phi[i],w[i]);
		}
		delete[] A;
		delete[] k;
		delete[] theta;
		delete[] phi;
		delete[] w;

		fclose(f);
	}
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Periodic_HeightField::loadSpectrum(const char* filename)
{
	FILE *f = fopen(filename,"r");
	if(f!=NULL){
		uint nbFunc;
		int nb = fscanf(f,"%d\n",&nbFunc);

		if(nb>0 && nbFunc>0){

			this->nbFunc = nbFunc;
			printf("nbFunc:%d\n",nbFunc);

			if(m_A!=NULL) freeArray(m_A);
			if(m_k!=NULL) freeArray(m_k);
			if(m_theta!=NULL) freeArray(m_theta);
			if(m_phi!=NULL) freeArray(m_phi);
			if(m_w!=NULL) freeArray(m_w);

			double *A, *k, *theta, *phi, *w;
			A = new double[nbFunc];
			k = new double[nbFunc];
			theta = new double[nbFunc];
			phi = new double[nbFunc];
			w = new double[nbFunc];

			for(uint i=0;i<nbFunc;i++){
				nb = fscanf(f,"A:%lf k:%lf theta:%lf phi:%lf w:%lf \n",&A[i],&k[i],&theta[i],&phi[i],&w[i]);
				printf("A:%f k:%f theta:%f phi:%f w:%f\n",A[i],k[i],theta[i],phi[i],w[i]);
			}
		
			allocateArray((void**)&m_A, nbFunc*sizeof(double));
			allocateArray((void**)&m_k, nbFunc*sizeof(double));
			allocateArray((void**)&m_theta, nbFunc*sizeof(double));
			allocateArray((void**)&m_phi, nbFunc*sizeof(double));
			allocateArray((void**)&m_w, nbFunc*sizeof(double));

			copyArrayToDevice(m_A, A, 0, nbFunc*sizeof(double));
			copyArrayToDevice(m_k, k, 0, nbFunc*sizeof(double));
			copyArrayToDevice(m_theta, theta, 0, nbFunc*sizeof(double));
			copyArrayToDevice(m_phi, phi, 0, nbFunc*sizeof(double));
			copyArrayToDevice(m_w, w, 0, nbFunc*sizeof(double));

			delete[] A;
			delete[] k;
			delete[] theta;
			delete[] phi;
			delete[] w;
		}
		fclose(f);
	}
}
/*****************************************************************************************************/
/*****************************************************************************************************/
}
/*****************************************************************************************************/
/*****************************************************************************************************/
