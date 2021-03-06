#include <Quadrilateral.h>
#include <GL/gl.h>

/**********************************************************************************/
/**********************************************************************************/
namespace Utils {
/**********************************************************************************/
/**********************************************************************************/
Quadrilateral::Quadrilateral():Face()
{
}
/**********************************************************************************/
Quadrilateral::Quadrilateral(vector<Vector3> vertexs):Face(vertexs)
{
	assert(vertexs.size()==4);
}
/**********************************************************************************/
Quadrilateral::Quadrilateral(Vector3 V0, Vector3 V1, Vector3 V2, Vector3 V3):Face()
{
	setVertex(V0); setVertex(V1); setVertex(V2); setVertex(V3);
	calculateNormale();
}
/**********************************************************************************/
Quadrilateral::Quadrilateral(const Quadrilateral& F):Face(F)
{
	assert(F.vertexs.size()==4);
	setVertexs(F.vertexs);
	setNormale(F.N);
}
/**********************************************************************************/
Quadrilateral::~Quadrilateral()
{
	this->vertexs.clear();
}
/**********************************************************************************/
/**********************************************************************************/
void Quadrilateral::create(Vector3 origin, float length, float width,Vector3 orientation)
{
	Vector3 V0, V1, V2, V3;
	if(orientation.x() == -1 || orientation.x() == 1){
       		 V0 = Vector3(origin.x(),origin.y()-(length/2),origin.z()-(width/2));
		 V1 = Vector3(origin.x(),origin.y()+(length/2),origin.z()-(width/2));
		 V2 = Vector3(origin.x(),origin.y()+(length/2),origin.z()+(width/2));
		 V3 = Vector3(origin.x(),origin.y()-(length/2),origin.z()+(width/2));
		 setVertex(V0);
		 setVertex(V1);
		 setVertex(V2);
		 setVertex(V3);
		 setNormale(Vector3(orientation.x(),0,0));
	}
        else {
		if(orientation.y() == -1 || orientation.y() == 1){
	       		 V0 = Vector3(origin.x()-(length/2),origin.y(),origin.z()-(width/2));
			 V1 = Vector3(origin.x()+(length/2),origin.y(),origin.z()-(width/2));
			 V2 = Vector3(origin.x()+(length/2),origin.y(),origin.z()+(width/2));
			 V3 = Vector3(origin.x()-(length/2),origin.y(),origin.z()+(width/2));
			 setVertex(V0);
		 	 setVertex(V1);
		  	 setVertex(V2);
		 	 setVertex(V3);
			 setNormale(Vector3(0,orientation.y(),0));
		}
		else {
			if(orientation.z() == -1 || orientation.z() == 1){
       		 		V0 = Vector3(origin.x()-(length/2),origin.y()-(width/2),origin.z());
		 		V1 = Vector3(origin.x()+(length/2),origin.y()-(width/2),origin.z());
		 		V2 = Vector3(origin.x()+(length/2),origin.y()+(width/2),origin.z());
		 		V3 = Vector3(origin.x()-(length/2),origin.y()+(width/2),origin.z());
		 		setVertex(V0);
		 		setVertex(V1);
		  		setVertex(V2);
		 		setVertex(V3);
		 		setNormale(Vector3(0,0,orientation.z()));
			}
		}
	}
}
/**********************************************************************************/
/**********************************************************************************/
}
/**********************************************************************************/
/**********************************************************************************/
