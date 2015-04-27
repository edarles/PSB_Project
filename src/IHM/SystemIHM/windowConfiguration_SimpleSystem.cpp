#include <windowConfiguration_SimpleSystem.h>
#include <SimpleSystem.h>

/************************************************************************************************************************************************/
/************************************************************************************************************************************************/
WindowConfiguration_SimpleSystem::WindowConfiguration_SimpleSystem(Window* parent, QVBoxLayout* mainLayoutParent, QHBoxLayout* layoutRightParent,
								   GLWidget *widget)
{
	this->parent = parent;
	this->mainLayoutParent = mainLayoutParent;
	this->layoutRightParent = layoutRightParent;

	gravityLabel = new QLabel("Gravity",this);
	gravity = new QDoubleSpinBox();
	gravity->setMinimum(0);
	gravity->setMaximum(1000);
	gravity->setValue(9.81);
	connect(gravity, SIGNAL(valueChanged(double)), this, SLOT(changeGravityData(double))); 
	
	deltaTimeLabel = new QLabel("Time step",this);
 	deltaTime = new QDoubleSpinBox();
	deltaTime->setMinimum(0.001);
	deltaTime->setMaximum(1.0);
	deltaTime->setValue(0.01);
	connect(deltaTime, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	QVBoxLayout *layout = new QVBoxLayout();
	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(gravityLabel,0,0);
	grid1->addWidget(gravity,0,1); 
	layout->addLayout(grid1);
	
	QGridLayout *grid3 = new QGridLayout();
	grid3->addWidget(deltaTimeLabel,0,0);
	grid3->addWidget(deltaTime,0,1);
	layout->addLayout(grid3);

	QGridLayout *grid4 = new QGridLayout();
	buttonOK = new QPushButton(tr("OK"),this);
	connect(buttonOK, SIGNAL(clicked()), this , SLOT(accept()));
	buttonCancel = new QPushButton(tr("Cancel"),this);
        connect(buttonCancel, SIGNAL(clicked()), this , SLOT(cancel()));
	grid4->addWidget(buttonOK,0,0);
	grid4->addWidget(buttonCancel,0,1);
	
	layout->addLayout(grid4);

	setLayout(layout);
	setWindowTitle("Configuration");

	this->glWidget = widget;
	SimpleSystem *system = new SimpleSystem();
        ForceExt_Constante *F = new ForceExt_Constante(Vector3(0,-1,0),gravity->value());
	system->addForce(F);
	system->setDt(deltaTime->value());

	glWidget->setSystem(system);
	
	page = NULL;
}
/****************************************************************************/
WindowConfiguration_SimpleSystem::~WindowConfiguration_SimpleSystem()
{
}
/****************************************************************************/
/****************************************************************************/
void WindowConfiguration_SimpleSystem::accept()
{
	glWidget->initSystem();
	glWidget->draw();
	close();
}
/****************************************************************************/
void WindowConfiguration_SimpleSystem::cancel()
{
	SimpleSystem *system = (SimpleSystem*) glWidget->getSystem();
	delete(system);
	close();
}
/****************************************************************************/
/****************************************************************************/
QSize WindowConfiguration_SimpleSystem::sizeHint() const
{
	return QSize(300,100);
}
/****************************************************************************/
/****************************************************************************/	
void WindowConfiguration_SimpleSystem::changeData(double d)
{
	glWidget->getSystem()->setDt(deltaTime->value());
}
/****************************************************************************/	
void WindowConfiguration_SimpleSystem::changeGravityData(double d)
{
	ForceExt_Constante *F = (ForceExt_Constante*) glWidget->getSystem()->getForcesExt()->getForce(0);
	if(F!=NULL) delete(F);
	F = new ForceExt_Constante(Vector3(0,-1,0),gravity->value());
	glWidget->getSystem()->getForcesExt()->setForce(F,0);
}
/****************************************************************************/
/****************************************************************************/
