#include <windowConfiguration_SPHSystem.h>
#include <SphSystem.h>

/*****************************************************************************************/
/*****************************************************************************************/
WindowConfiguration_SPHSystem::WindowConfiguration_SPHSystem(GLWidget *widget)
{
	gravityLabel = new QLabel("Gravity",this);
	gravity = new QDoubleSpinBox();
	gravity->setMinimum(0);
	gravity->setMaximum(1000);
	gravity->setValue(9.81);
	connect(gravity, SIGNAL(valueChanged(double)), this, SLOT(changeGravityData(double))); 
	
	deltaTimeLabel = new QLabel("Time step",this);
 	deltaTime = new QDoubleSpinBox();
	deltaTime->setMinimum(0.00001);
	deltaTime->setMaximum(10.0);
	deltaTime->setValue(0.01);
	connect(deltaTime, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 
      
	QVBoxLayout *layout = new QVBoxLayout();

	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(gravityLabel,0,0);
	grid1->addWidget(gravity,0,1); 
	layout->addLayout(grid1);

	QGridLayout *grid4 = new QGridLayout();
	grid4->addWidget(deltaTimeLabel,0,0);
	grid4->addWidget(deltaTime,0,1);
	layout->addLayout(grid4);

	QGridLayout *grid10 = new QGridLayout();
	buttonOK = new QPushButton(tr("OK"),this);
	connect(buttonOK, SIGNAL(clicked()), this , SLOT(accept()));
	buttonCancel = new QPushButton(tr("Cancel"),this);
        connect(buttonCancel, SIGNAL(clicked()), this , SLOT(cancel()));
	grid10->addWidget(buttonOK,0,0);
	grid10->addWidget(buttonCancel,0,1);
	layout->addLayout(grid10);

	setLayout(layout);
	setWindowTitle("Configuration");

	this->glWidget = widget;
	SPHSystem *system = new SPHSystem();
	system->setDt(deltaTime->value());
        ForceExt_Constante *F = new ForceExt_Constante(Vector3(0,-1,0),gravity->value());
	system->addForce(F);
	
	glWidget->setSystem(system);

}
/*****************************************************************************************/
/*****************************************************************************************/
WindowConfiguration_SPHSystem::~WindowConfiguration_SPHSystem()
{
}
/*****************************************************************************************/
/*****************************************************************************************/
void WindowConfiguration_SPHSystem::accept()
{
	glWidget->initSystem();
	glWidget->draw(); 
	close();
}
/*****************************************************************************************/
void WindowConfiguration_SPHSystem::cancel()
{
	SPHSystem *system = (SPHSystem*) glWidget->getSystem();
	delete(system);
	close();
}
/*****************************************************************************************/
/*****************************************************************************************/
QSize WindowConfiguration_SPHSystem::sizeHint() const
{
     return QSize(300, 100);
}
/****************************************************************************/
/****************************************************************************/	
void WindowConfiguration_SPHSystem::changeData(double d)
{
	glWidget->getSystem()->setDt(deltaTime->value());
}
/****************************************************************************/	
void WindowConfiguration_SPHSystem::changeGravityData(double d)
{
	ForceExt_Constante *F = (ForceExt_Constante*) glWidget->getSystem()->getForcesExt()->getForce(0);
	if(F!=NULL) delete(F);
	F = new ForceExt_Constante(Vector3(0,-1,0),gravity->value());
	glWidget->getSystem()->getForcesExt()->setForce(F,0);
}
/******************************************************************************************/
/******************************************************************************************/
