#include <ParticleLoader_Txt.h>
#include <stdio.h>

namespace Utils {

ParticleLoader_Txt::ParticleLoader_Txt():ParticleLoader()
{
}
ParticleLoader_Txt::~ParticleLoader_Txt()
{
}
vector<Particle*> ParticleLoader_Txt::load(const char *filename)
{
	FILE *f = fopen(filename,"r");
 	if(f!=NULL){
			vector<Particle*> particles;
    			int nbPos;
     			int nbLu = fscanf(f,"%d\n",&nbPos);
			for(int i=0;i<nbPos;i++){
				double x,y,z,vx,vy,vz,m,r,cx,cy,cz;
				nbLu = fscanf(f,"particle -p %lf %lf %lf -v %lf %lf %lf -m %lf -r %lf -c %lf %lf %lf\n",&x,&y,&z,&vx,&vy,&vz,&m,
					      &r,&cx,&cy,&cz);
				Particle *p = new Particle(Vector3(x,y,z),Vector3(vx,vy,vz),m,r,Vector3(cx,cy,cz));
				particles.push_back(p);
			}
			fclose(f);
			return particles;
	}	
}

}
