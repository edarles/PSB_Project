#include <GL/glew.h>
#include <System.h>
#include <typeinfo>
#include <shaders.h>
#if defined(__APPLE__) || defined(__MACOSX)
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

/*********************************************************************************************/
/*********************************************************************************************/
System::System()
{ 
 //glutInitDisplayMode(GLUT_RGBA | GLUT_3_2_CORE_PROFILE);
 m_program = _compileProgram(vertexShader, spherePixelShader);
 collision = new Collision();
 emitters = new Emitters();
 FExt = new ForcesExt();
 dt = 0.01;
 t = 0;
}
/*********************************************************************************************/
System::~System()
{
 glDeleteProgram(m_program);
 //#pragma omp parallel for
 for(unsigned int i=0;i<particles.size();i++)
	delete(particles[i]);
 particles.clear();
 
 delete(collision);
 delete(emitters);
 delete(FExt);
}
/*********************************************************************************************/
/*********************************************************************************************/
double System::getDt()
{
	return dt;
}
/*********************************************************************************************/
void System::setDt(double dt)
{
	this->dt = dt;
}
/*********************************************************************************************/
/*********************************************************************************************/
ObjectCollision* System::getObjectCollision(unsigned int i)
{
	return collision->getObject(i);
}
/*********************************************************************************************/
vector<ObjectCollision*> System::getObjectsCollision()
{
	return collision->getObjects();
}
/*********************************************************************************************/
/*********************************************************************************************/
void System::addObjectCollision(ObjectCollision *O)
{
	collision->setObject(O);
}
/*********************************************************************************************/
void System::removeLastObjectCollision()
{ 
	collision->removeLastObject();
}
/*********************************************************************************************/
Emitters*	System::getEmitters()
{
	return emitters;
}
/*********************************************************************************************/
ForcesExt* System::getForcesExt()
{
	return FExt;
}
/*********************************************************************************************/
/*********************************************************************************************/
void System::addForce(ForceExt* F)
{
	FExt->addForce(F);
}
/*********************************************************************************************/
void	System::addEmitter(Emitter* E)
{
	emitters->addEmitter(E);
}
/*********************************************************************************************/
vector<Particle*> System::getParticles()
{
	return particles;
}
/*********************************************************************************************/
void System::setParticles(vector<Particle*> particles)
{
	#pragma omp parallel for
	for(unsigned int i=0;i<particles.size();i++){
		this->particles[i] = new Particle();
		this->particles[i] = particles[i];
	}
}
/*********************************************************************************************/
/*********************************************************************************************/
void System::displayParticles(ParticleDisplay mode, Vector3 color)
{
   glColor3f(1.0,1.0,1.0);
   double scaleV = 0.1;
   switch(mode)
   {
       // std::stringstream fps_text, mode_text, grid_text, gpumem_text,
       // memtrans_text, mflops_text;
        //fps_text << "FPS: " << fps << " Frames total: " << frame;
      //  mode_text << "Display Mode: ";
      
	default:
	case POINTS:
			glPointSize(1.0);
			glColor3f(1,1,1);
			glEnableClientState(GL_VERTEX_ARRAY); 
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(4, GL_DOUBLE, 0, m_hColors);
        		glVertexPointer(3, GL_DOUBLE, 0, m_hPos[1]);
			glDrawArrays(GL_POINTS, 0, particles.size());
			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);
			
		break;
	case SPHERES:
			glEnable(GL_POINT_SPRITE_ARB);
			
			//glEnable(GL_BLEND);                                // Allow Transparency
   			//glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);  // how transparency acts
       			glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
        		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
        		glDepthMask(GL_TRUE);
        		glEnable(GL_DEPTH_TEST);
			
        		glUseProgram(m_program);
        		glUniform1f( glGetUniformLocation(m_program, "pointScale"), 768 / tanf(60*0.5f*(float)M_PI/180.0f) );
        		glUniform1f( glGetUniformLocation(m_program, "pointRadius"), particles[0]->getParticleRadius()/1.5);
			glUniform1f( glGetUniformLocation(m_program, "near"), 0.000000001 );
			glUniform1f( glGetUniformLocation(m_program, "far"), 0.00001 );

        		glPointSize(1.0);
			glColor3f(1,1,1);
			glEnableClientState(GL_VERTEX_ARRAY); 
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(4, GL_DOUBLE, 0, m_hColors);
        		glVertexPointer(3, GL_DOUBLE, 0, m_hPos[1]);
			glDrawArrays(GL_POINTS, 0, particles.size());
			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);

        		glUseProgram(0);
			glDisable(GL_BLEND);
        		glDisable(GL_POINT_SPRITE_ARB);
/*
			for(uint i=0;i<particles.size();i++){
				glColor3f(m_hColors[i*4],m_hColors[i*4+1],m_hColors[i*4+2]);
				glTranslatef(m_hPos[1][i*3],m_hPos[1][i*3+1],m_hPos[1][i*3+2]);
				glutSolidSphere(particles[i]->getParticleRadius()/1.5,20,20);
				glTranslatef(-m_hPos[1][i*3],-m_hPos[1][i*3+1],-m_hPos[1][i*3+2]);
                     	}
			
*/
			break;
	//case SURFACE:
	//	break;
	
   }
   /*glLineWidth(0.1);
			for(uint i=0;i<particles.size();i++){
				glColor3f(m_hColors[i*4],m_hColors[i*4+1],m_hColors[i*4+2]);
				glBegin(GL_LINES);
				
				glVertex3f(m_hPos[1][i*3],m_hPos[1][i*3+1],m_hPos[1][i*3+2]);
				
				glVertex3f(m_hPos[1][i*3]+m_hVel[1][i*3]*scaleV,m_hPos[1][i*3+1]+m_hVel[1][i*3+1]*scaleV,m_hPos[1][i*3+2]+m_hVel[1][i*3+2]*scaleV);
				glEnd();
			}
	*/
}
/*********************************************************************************************/
/*********************************************************************************************/
void System::displayParticlesByField(uint field)
{
}
/*********************************************************************************************/
/*********************************************************************************************/
void System::displayCollisions(GLenum face, GLenum raster, Vector3 colorObject, Vector3 colorNormales, bool normales)
{
     if(collision!=NULL)
	collision->display(face, raster, colorObject, colorNormales, normales);
}
/*********************************************************************************************/
void System::displayEmitters(Vector3 color)
{
	emitters->display(color);
}
/*********************************************************************************************/
/*********************************************************************************************/
GLuint System::_compileProgram(const char *vsource, const char *fsource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);
    
    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        char temp[256];
	printf("succes:%d\n",success);
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }
    return program;
}


