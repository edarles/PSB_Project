#include <MeshCollision.h>
#include <MeshCollision.cuh>
/******************************************************************************************/
/******************************************************************************************/
MeshCollision::MeshCollision():ObjectCollision(), Mesh()
{
}
/******************************************************************************************/
MeshCollision::MeshCollision(float elast, float friction, bool is_container)
	      :ObjectCollision(elast, friction, is_container),Mesh()
{
}
/******************************************************************************************/
MeshCollision::MeshCollision(const MeshCollision& O):ObjectCollision(O),Mesh(O)
{
}
/******************************************************************************************/
MeshCollision::~MeshCollision()
{
	freeArray(m_F[0]); 
	freeArray(m_F[1]); 
	freeArray(m_F[2]); 
	freeArray(m_N[0]); 
}
/******************************************************************************************/
/******************************************************************************************/
void MeshCollision::collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		            float dt, int nbBodiesP)
{
	collisionMesh_CUDA(oldPos, newPos, oldVel, newVel, radiusParticle, dt, getNbFaces(), m_F[0], m_F[1], m_F[2], m_N[0], 
			   getElast(), nbBodiesP);
}
/******************************************************************************************/
/******************************************************************************************/
void MeshCollision::storeFacesOnGPU()
{
	m_F[0] = m_F[1] = m_F[2] = 0;
	m_N[0] = 0;

	unsigned int memSize = sizeof(float) * 3 * faces.size();
	float* fA = new float[faces.size()*3];
    	float* fB = new float[faces.size()*3];
	float* fC = new float[faces.size()*3];
    	float* N = new float[faces.size()*3];

   	memset(fA, 0, memSize);
    	memset(fB, 0, memSize);
	memset(fC, 0, memSize);
	memset(N, 0, memSize);

	for(unsigned int i=0;i<faces.size();i++){
		Triangle f = faces[i];

		if(is_container){
			f.setNormale(-f.getNormale());
			faces[i].setNormale(f.getNormale());
		}
		fA[(i*3)] = f.getVertex(0).x();
		fA[(i*3)+1] = f.getVertex(0).y();
		fA[(i*3)+2] = f.getVertex(0).z();

		fB[(i*3)] = f.getVertex(1).x();
		fB[(i*3)+1] = f.getVertex(1).y();
		fB[(i*3)+2] = f.getVertex(1).z();

		fC[(i*3)] = f.getVertex(2).x();
		fC[(i*3)+1] = f.getVertex(2).y();
		fC[(i*3)+2] = f.getVertex(2).z();

		N[(i*3)] = f.getNormale().x();
		N[(i*3)+1] = f.getNormale().y();
		N[(i*3)+2] = f.getNormale().z();
	}
	allocateArray((void**)&m_F[0], memSize);
    	allocateArray((void**)&m_F[1], memSize);
	allocateArray((void**)&m_F[2], memSize);
	allocateArray((void**)&m_N[0], memSize);

	copyArrayToDevice(m_F[0],fA,0,faces.size()*3*sizeof(float));
	copyArrayToDevice(m_F[1],fB,0,faces.size()*3*sizeof(float));
	copyArrayToDevice(m_F[2],fC,0,faces.size()*3*sizeof(float));
	copyArrayToDevice(m_N[0],N,0,faces.size()*3*sizeof(float));

	free(fA); free(fB); free(fC); free(N);
}
/******************************************************************************************/
/******************************************************************************************/
void MeshCollision::display(GLenum mode, GLenum raster, Vector3 color)
{
	glPolygonMode(mode,raster);
	Mesh::display(color);
}
/******************************************************************************************/
void MeshCollision::displayNormales(Vector3 color)
{
	Mesh::displayNormale(color);
}
/******************************************************************************************/
/******************************************************************************************/
