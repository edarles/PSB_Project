#include <SurfaceSPH.h>
#include <MarchingCube.cuh>
#include <algorithm>

/***********************************************************************************************/
/***********************************************************************************************/
SurfaceSPH::SurfaceSPH():Surface()
{
	indexs_cpu.clear();
	vertexs_cpu.clear();
	normales_cpu.clear();
}
/***********************************************************************************************/
SurfaceSPH::SurfaceSPH(Vector3 color, double transparency):Surface(color,transparency)
{
	indexs_cpu.clear();
	vertexs_cpu.clear();
	normales_cpu.clear();
}
/***********************************************************************************************/
SurfaceSPH::~SurfaceSPH()
{
	indexs_cpu.clear();
	vertexs_cpu.clear();
	normales_cpu.clear();
}
/***********************************************************************************************/
/***********************************************************************************************/
bool SurfaceSPH::extract(System *S, double isoLevel, uint step)
{
	float sizeCell = 0.03;	
	double scale = 0.65;

	assert(typeid(*S)==typeid(SPHSystem));
	SPHSystem *s = (SPHSystem*) S;
	/*******************************************************************************/
	// Calculate Min and Max points grid, length, depth and width of grid
	/*******************************************************************************/
	double xMin = 10000; double yMin = 10000; double zMin = 10000;
        double xMax = -10000; double yMax = -10000; double zMax = -10000;
	for(uint i=0;i<s->particles.size();i++){
		double x = s->m_hPos[1][i*3];
		double y = s->m_hPos[1][i*3+1];
		double z = s->m_hPos[1][i*3+2];
		double r = 2*(1/scale)*s->m_hInteractionRadius[i];
		if(xMin>(x-r)) xMin = x-r; if(yMin>(y-r)) yMin = y-r; if(zMin>(z-r)) zMin = z-r;
		if(xMax<(x+r)) xMax = x+r; if(yMax<(y+r)) yMax = y+r; if(zMax<(z+r)) zMax = z+r;
	}
	Vector3 Min(xMin,yMin,zMin);
	Vector3 Max(xMax,yMax,zMax);
	Vector3 center((Min.x()+Max.x())/2,(Min.y()+Max.y())/2,(Min.z()+Max.z())/2);
	
	double l = fabs(Max.x()-Min.x());
	double w = fabs(Max.y()-Min.y());
	double d = fabs(Max.z()-Min.z());
	printf("l:%f w:%f d:%f\n",l,w,d);

	uint nbPosX = ceil(l/sizeCell);
	uint nbPosY = ceil(w/sizeCell);
	uint nbPosZ = ceil(d/sizeCell);

	printf("nbPosX:%d nbPosY:%d nbPosZ:%d\n",nbPosX,nbPosY,nbPosZ);
	/*******************************************************************************/	
	// Allocate structures
	/*******************************************************************************/
	uint size = nbPosX + nbPosX*nbPosY + nbPosX*nbPosY*nbPosZ;
	/*******************************************************************************/
	// CPU structures
	/*******************************************************************************/
	double *grid, *vertexs, *vertex2, *normales, *normales2;
	int *indexs, *nbIndex;
	vertexs = new double[size*3*nbVertexMax_perCell];
	normales = new double[size*3*nbVertexMax_perCell];
	indexs = new int[size*nbIndexMax_perCell];
	grid = new double[4*size];
	nbIndex = new int[size];
	vertex2 = new double[3*nbVertexMax];
	normales2 = new double[3*nbVertexMax];
	/*******************************************************************************/
	// GPU structures
	/*******************************************************************************/
	double *m_grid, *m_vertex, *m_vertex2, *m_normales, *m_normales2;
	int *m_nbIndex, *m_index;
	allocateArray((void**)&m_grid,sizeof(double)*4*size);
	allocateArray((void**)&m_vertex,sizeof(double)*3*size*nbVertexMax_perCell);
	allocateArray((void**)&m_normales,sizeof(double)*3*size*nbVertexMax_perCell);
	allocateArray((void**)&m_index,sizeof(int)*size*nbIndexMax_perCell);
	allocateArray((void**)&m_nbIndex,sizeof(int)*size);
	allocateArray((void**)&m_vertex2,sizeof(double)*3*nbVertexMax);
	allocateArray((void**)&m_normales2,sizeof(double)*3*nbVertexMax);
	/*******************************************************************************/
	// Create Grid
	/*******************************************************************************/
	createGrid_MarchingCube_CUDA(center, l, w, d, sizeCell, nbPosX, nbPosY, nbPosZ, m_grid, m_nbIndex);

	/*******************************************************************************/
	// Calculate iso-values in cells
	/*******************************************************************************/
	s->evaluateIso(m_grid, nbPosX, nbPosY, nbPosZ, scale);

	/*******************************************************************************/
	// Polygonize
	/*******************************************************************************/
	polygonize_MarchingCube_CUDA(m_grid, nbPosX, nbPosY, nbPosZ, isoLevel, 
				     m_nbIndex, m_vertex, m_normales, m_index);
	
	/*******************************************************************************/
	// GPU -> CPU transferts	
	/*******************************************************************************/
	copyArrayFromDevice(grid,m_grid,0,sizeof(double)*4*size);
	copyArrayFromDevice(vertexs,m_vertex,0,sizeof(double)*3*size*nbVertexMax_perCell);
	copyArrayFromDevice(indexs,m_index,0,sizeof(int)*size*nbIndexMax_perCell);
	copyArrayFromDevice(nbIndex,m_nbIndex,0,sizeof(int)*size);
	copyArrayFromDevice(normales,m_normales,0,sizeof(double)*3*size*nbVertexMax_perCell);

	/*******************************************************************************/
	// Transferts each vertexs in cells into a tab structure
	/********00***********************************************************************/
	indexs_cpu.clear();
	vertexs_cpu.clear();
	normales_cpu.clear();
	uint nbV = 0;
	for(uint i=0;i<nbPosX;i++){
		for(uint j=0;j<nbPosY;j++){
			for(uint k=0;k<nbPosZ;k++){
				uint indexC = i + j*nbPosX + k*nbPosX*nbPosY;
				uint nb = nbIndex[indexC];
				if(nb>0){
					uint nbT = 0;
					for(uint n=0;n<nb;n++){
						int ind = indexs[indexC*nbIndexMax_perCell+n];
						Vector3 P1;
						P1.setX(vertexs[(indexC*nbVertexMax_perCell + ind)*3]);
						P1.setY(vertexs[(indexC*nbVertexMax_perCell + ind)*3+1]);
						P1.setZ(vertexs[(indexC*nbVertexMax_perCell + ind)*3+2]);
						vector<Vector3>::const_iterator found = std::find(vertexs_cpu.begin(), vertexs_cpu.end(), P1);
						if(found==vertexs_cpu.end()){
							indexs_cpu.push_back(nbV);
							vertexs_cpu.push_back(P1);
							vertex2[nbV*3]=P1.x(); 
							vertex2[nbV*3+1]=P1.y(); 
							vertex2[nbV*3+2]=P1.z();	
							nbV++;
						}
						else {
							//printf("vertex trouve à la place:%d\n", found - vertexs_cpu.begin());
							//printf("V0:%f %f %f V1:%f %f %f\n",P1.x(),P1.y(),P1.z(),
							//vertexs_cpu[found - vertexs_cpu.begin()].x(),
							//vertexs_cpu[found - vertexs_cpu.begin()].y(),
							//vertexs_cpu[found - vertexs_cpu.begin()].z());
							indexs_cpu.push_back(found - vertexs_cpu.begin());
						}					
					}
				}
			}
		}
	}
	/*******************************************************************************/
	copyArrayToDevice(m_vertex2,vertex2,0,sizeof(double)*3*nbVertexMax);
	s->computeNormales_Vertex(m_vertex2, m_normales2, scale, nbV);
	copyArrayFromDevice(normales2,m_normales2,0,sizeof(double)*3*nbVertexMax);
	for(uint i=0;i<nbV;i++){
		Vector3 N;
		N.setX(normales2[i*3]); N.setY(normales2[i*3+1]); N.setZ(normales2[i*3+2]);
		normales_cpu.push_back(N);
	}
	/*******************************************************************************/
	// Desallocate structures
	/*******************************************************************************/
	delete[] vertexs;
	delete[] normales;
	delete[] indexs;
	delete[] grid;
	delete[] nbIndex;
	delete[] vertex2;
	delete[] normales2;

	freeArray(m_grid);
	freeArray(m_vertex);
	freeArray(m_index);
	freeArray(m_nbIndex);
	freeArray(m_normales);
	freeArray(m_vertex2);
	freeArray(m_normales2);

	return true;
}
/***********************************************************************************************/
/***********************************************************************************************/
void SurfaceSPH::draw()
{
	glColor3f(1,1,1);
	double d = 0.1;
	if(indexs_cpu.size()>0){
		glShadeModel(GL_SMOOTH);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		for(uint i=0;i<indexs_cpu.size()-3;i+=3){
			glColor3f(1,1,1);
			glBegin(GL_TRIANGLES);
			glVertex3f(vertexs_cpu[indexs_cpu[i]].x(),vertexs_cpu[indexs_cpu[i]].y(),vertexs_cpu[indexs_cpu[i]].z());
			glNormal3f(normales_cpu[indexs_cpu[i]].x(),normales_cpu[indexs_cpu[i]].y(),normales_cpu[indexs_cpu[i]].z());
			glVertex3f(vertexs_cpu[indexs_cpu[i+1]].x(),vertexs_cpu[indexs_cpu[i+1]].y(),vertexs_cpu[indexs_cpu[i+1]].z());
			glNormal3f(normales_cpu[indexs_cpu[i+1]].x(),normales_cpu[indexs_cpu[i+1]].y(),normales_cpu[indexs_cpu[i+1]].z());
			glVertex3f(vertexs_cpu[indexs_cpu[i+2]].x(),vertexs_cpu[indexs_cpu[i+2]].y(),vertexs_cpu[indexs_cpu[i+2]].z());
			glNormal3f(normales_cpu[indexs_cpu[i+2]].x(),normales_cpu[indexs_cpu[i+2]].y(),normales_cpu[indexs_cpu[i+2]].z());
			glEnd();

			glColor3f(1,0,0);
			glBegin(GL_LINES);
			glVertex3f(vertexs_cpu[indexs_cpu[i]].x(),vertexs_cpu[indexs_cpu[i]].y(),vertexs_cpu[indexs_cpu[i]].z());
			glVertex3f(vertexs_cpu[indexs_cpu[i]].x()+normales_cpu[indexs_cpu[i]].x()*d,
				   vertexs_cpu[indexs_cpu[i]].y()+normales_cpu[indexs_cpu[i]].y()*d,
			           vertexs_cpu[indexs_cpu[i]].z()+normales_cpu[indexs_cpu[i]].z()*d);
			glEnd();
			glBegin(GL_LINES);
			glVertex3f(vertexs_cpu[indexs_cpu[i+1]].x(),vertexs_cpu[indexs_cpu[i+1]].y(),vertexs_cpu[indexs_cpu[i+1]].z());
			glVertex3f(vertexs_cpu[indexs_cpu[i+1]].x()+normales_cpu[indexs_cpu[i+1]].x()*d,
				   vertexs_cpu[indexs_cpu[i+1]].y()+normales_cpu[indexs_cpu[i+1]].y()*d,
			           vertexs_cpu[indexs_cpu[i+1]].z()+normales_cpu[indexs_cpu[i+1]].z()*d);
			glEnd();
			glBegin(GL_LINES);
			glVertex3f(vertexs_cpu[indexs_cpu[i+2]].x(),vertexs_cpu[indexs_cpu[i+2]].y(),vertexs_cpu[indexs_cpu[i+2]].z());
			glVertex3f(vertexs_cpu[indexs_cpu[i+2]].x()+normales_cpu[indexs_cpu[i+2]].x()*d,
				   vertexs_cpu[indexs_cpu[i+2]].y()+normales_cpu[indexs_cpu[i+2]].y()*d,
			           vertexs_cpu[indexs_cpu[i+2]].z()+normales_cpu[indexs_cpu[i+2]].z()*d);
			glEnd();

		}
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}		
}
/***********************************************************************************************/
/***********************************************************************************************/
bool SurfaceSPH::exportOBJ(const char* filename)
{
	if(indexs_cpu.size()>0){
		// create diffuse material
		FILE* f1 = fopen("fluid.mtl","w");
		fprintf(f1,"# Blender3D MTL File: <memory>\n");
		fprintf(f1,"# Material Count: 1\n");
		fprintf(f1,"newmtl fluid_mtl\n");
		fprintf(f1,"Ka 0.000000 0.000000 0.000000\n");
		fprintf(f1,"Kd 0.8 0.8 0.8\n");
		fprintf(f1,"Ks 0.5 0.5 0.5\n");
		fprintf(f1,"Ns 100\n");
		fprintf(f1,"Ni 0.000000\n");
		fclose(f1);

		//store triangles in OBJ file
		FILE* f = fopen(filename,"w");
		fprintf(f,"# Blender v2.62 (sub 0) OBJ File: ''\n");
		fprintf(f,"# www.blender.org\n");
		fprintf(f,"mtllib fluid.mtl\n");
		fprintf(f,"o isosurface\n");

		for(unsigned long i=0;i<vertexs_cpu.size();i++)
			//fprintf(f,"v %f %f %f\n",vertexs_cpu[i].x(),vertexs_cpu[i].z(),vertexs_cpu[i].y());
			fprintf(f,"v %.5f %.5f %.5f\n",vertexs_cpu[i].x()*2,vertexs_cpu[i].z()*2,vertexs_cpu[i].y()*2+1);
		for(unsigned long i=0;i<normales_cpu.size();i++)
			fprintf(f,"vn %.5f %.5f %.5f\n",normales_cpu[i].x(),normales_cpu[i].z(),normales_cpu[i].y());
		
		fprintf(f,"usemtl fluid_mtl\n");
		fprintf(f,"s off\n");
		fprintf(f,"f\n");
		for(unsigned long i=0;i<indexs_cpu.size()-3;i+=3){
			uint i1 = indexs_cpu[i];
			uint i2 = indexs_cpu[i+1];
			uint i3 = indexs_cpu[i+2];
			assert(i1<vertexs_cpu.size() && i2<vertexs_cpu.size() && i3<vertexs_cpu.size());
			fprintf(f,"f %d//%d %d//%d %d//%d\n",i1+1,i1+1,i2+1,i2+1,i3+1,i3+1);
		}
		fclose(f);
		return true;
	}
	return false;	
}

