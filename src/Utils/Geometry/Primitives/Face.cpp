#include <Face.h>
#include <GL/gl.h>

/*************************************************************************************/
/*************************************************************************************/
namespace Utils {

/*************************************************************************************/
/*************************************************************************************/
Face::Face():ObjectGeo()
{
	vertexs.clear();
	N = Vector3(0,0,0);
}
/*************************************************************************************/
Face::Face(vector<Vector3> vertexs):ObjectGeo()
{
	this->vertexs.clear();
	for(unsigned int i=0;i<vertexs.size();i++)
		this->vertexs.push_back(vertexs[i]);	
	calculateNormale();
}
/*************************************************************************************/
Face::Face(const Face& F):ObjectGeo()
{
	this->vertexs.clear();
	for(unsigned int i=0;i<F.vertexs.size();i++)
		this->vertexs.push_back(F.vertexs[i]);
	this->N = F.N;
}
/*************************************************************************************/
Face::~Face()
{
	this->vertexs.clear();
}
/*************************************************************************************/
/*************************************************************************************/
vector<Vector3> Face::getVertexs()
{
	return vertexs;
}
/*************************************************************************************/
Vector3 Face::getVertex(unsigned int i)
{
	assert(i<vertexs.size());
	return vertexs[i];
}
/*************************************************************************************/
Vector3 Face::getNormale()
{
	return N;
}
/*************************************************************************************/
/*************************************************************************************/
void Face::setVertex(unsigned int i, Vector3 V)
{
	assert(i<vertexs.size());
	vertexs[i] = V;
}
/*************************************************************************************/
void Face::setVertexs(vector<Vector3> vertexs)
{
	this->vertexs.clear();
	for(unsigned int i=0;i<vertexs.size();i++)
		this->vertexs.push_back(vertexs[i]);	
}
void Face::setVertex(Vector3 V)
{
	vertexs.push_back(V);
}
/*************************************************************************************/
void Face::setNormale(Vector3 N)
{
	this->N = N;
}
/*************************************************************************************/
/*************************************************************************************/
void Face::calculateNormale()
{
	Vector3 A = getVertex(0);
	Vector3 B = getVertex(1);
	Vector3 C = getVertex(2);

	Vector3 V1 = B-A;
	Vector3 V2 = C-B;
	N = Vector3(V1.y()*V2.z()-V1.z()*V2.y(),
		    V1.z()*V2.x()-V1.x()*V2.z(),
                    V1.x()*V2.y()-V1.y()*V2.x());

	N.makeUnitVector();
}
/*************************************************************************************/
/*************************************************************************************/
void Face::inverseNormale()
{
	N = -N;
}
/*************************************************************************************/
/*************************************************************************************/
void Face::display(Vector3 color)
{
	glLineWidth(2.0);
	glColor3f(color.x(),color.y(),color.z());
	glBegin(GL_POLYGON);
	for(unsigned int i=0;i<vertexs.size();i++){
	 	Vector3 v = vertexs[i];
		glVertex3f(v.x(),v.y(),v.z());
	}
	glEnd();
	glLineWidth(1.0);
	glColor3f(1.0,1.0,1.0);
}
/*************************************************************************************/
void Face::displayNormale(Vector3 color)
{
	glColor3f(color.x(),color.y(),color.z());
	Vector3 G(0,0,0);
	for(unsigned int i=0;i<vertexs.size();i++)
		G.setXYZ(G.x()+vertexs[i].x(),G.y()+vertexs[i].y(),G.z()+vertexs[i].z());
	G.setXYZ(G.x()/vertexs.size(),G.y()/vertexs.size(),G.z()/vertexs.size());

	glLineWidth(3.0);
	glBegin(GL_LINES);
	glVertex3f(G.x(),G.y(),G.z());
	glVertex3f(G.x()+N.x()*0.3,G.y()+N.y()*0.3,G.z()+N.z()*0.3);
	glEnd();
        glLineWidth(1.0);
	glColor3f(1.0,1.0,1.0);
}
/*************************************************************************************/
/*************************************************************************************/
}
/*************************************************************************************/
/*************************************************************************************/
