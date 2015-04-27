#include <windowConfiguration_PCI_SPHSystem.h>
#include <PciSphSystem.h>

/*****************************************************************************************/
/*****************************************************************************************/
WindowConfiguration_PCI_SPHSystem::WindowConfiguration_PCI_SPHSystem(GLWidget *widget)
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
	deltaTime->setDecimals(5);
	//deltaTime->setValue(0.01);
	deltaTime->setValue(0.001);
	connect(deltaTime, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 
      
	MinS = new QLabel("Min AABB grid",this);
	MinSX = new QDoubleSpinBox(); MinSY = new QDoubleSpinBox(); MinSZ = new QDoubleSpinBox();
	MinSX->setMinimum(-5); MinSY->setMinimum(-5); MinSZ->setMinimum(-5);
	MinSX->setMaximum(5); MinSY->setMaximum(5); MinSZ->setMaximum(5);
	MinSX->setValue(-1.0);   MinSY->setValue(-1.0); MinSZ->setValue(-1.0); 
	connect(MinSX, SIGNAL(valueChanged(double)), this, SLOT(changeMinS(double))); 
        connect(MinSY, SIGNAL(valueChanged(double)), this, SLOT(changeMinS(double))); 
        connect(MinSZ, SIGNAL(valueChanged(double)), this, SLOT(changeMinS(double))); 

	MaxS = new QLabel("Max AABB grid",this);
	MaxSX = new QDoubleSpinBox(); MaxSY = new QDoubleSpinBox(); MaxSZ = new QDoubleSpinBox();
	MaxSX->setMinimum(-5); MaxSY->setMinimum(-5); MaxSZ->setMinimum(-5);
	MaxSX->setMaximum(5); MaxSY->setMaximum(5); MaxSZ->setMaximum(5);
	MaxSX->setValue(1.0);   MaxSY->setValue(1.0); MaxSZ->setValue(1.0); 
	connect(MaxSX, SIGNAL(valueChanged(double)), this, SLOT(changeMaxS(double))); 
        connect(MaxSY, SIGNAL(valueChanged(double)), this, SLOT(changeMaxS(double))); 
        connect(MaxSZ, SIGNAL(valueChanged(double)), this, SLOT(changeMaxS(double))); 

	QVBoxLayout *layout = new QVBoxLayout();

	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(gravityLabel,0,0);
	grid1->addWidget(gravity,0,1); 
	layout->addLayout(grid1);

	QGridLayout *grid2 = new QGridLayout();
	grid2->addWidget(MinS,0,0);
	grid2->addWidget(MinSX,0,1); 
	grid2->addWidget(MinSY,0,2); 
	grid2->addWidget(MinSZ,0,3); 
	layout->addLayout(grid2);

	QGridLayout *grid3 = new QGridLayout();
	grid3->addWidget(MaxS,0,0);
	grid3->addWidget(MaxSX,0,1); 
	grid3->addWidget(MaxSY,0,2); 
	grid3->addWidget(MaxSZ,0,3); 
	layout->addLayout(grid3);

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
	PCI_SPHSystem *system = new PCI_SPHSystem();
	system->setDt(deltaTime->value());
	system->setMinS(Vector3(MinSX->value(),MinSY->value(),MinSZ->value()));
	system->setMaxS(Vector3(MaxSX->value(),MaxSY->value(),MaxSZ->value()));
        ForceExt_Constante *F = new ForceExt_Constante(Vector3(0,-1,0),gravity->value());
	system->addForce(F);
	
	glWidget->setSystem(system);

}
/*****************************************************************************************/
/*****************************************************************************************/
WindowConfiguration_PCI_SPHSystem::~WindowConfiguration_PCI_SPHSystem()
{
}
/*****************************************************************************************/
/*****************************************************************************************/
void WindowConfiguration_PCI_SPHSystem::accept()
{
	glWidget->initSystem();
	glWidget->draw(); 
	close();
}
/*****************************************************************************************/
void WindowConfiguration_PCI_SPHSystem::cancel()
{
	SPHSystem *system = (SPHSystem*) glWidget->getSystem();
	delete(system);
	close();
}
/*****************************************************************************************/
/*****************************************************************************************/
QSize WindowConfiguration_PCI_SPHSystem::sizeHint() const
{
     return QSize(300, 100);
}
/****************************************************************************/
/****************************************************************************/	
void WindowConfiguration_PCI_SPHSystem::changeData(double d)
{
	glWidget->getSystem()->setDt(deltaTime->value());
}
/****************************************************************************/	
void WindowConfiguration_PCI_SPHSystem::changeGravityData(double d)
{
	ForceExt_Constante *F = (ForceExt_Constante*) glWidget->getSystem()->getForcesExt()->getForce(0);
	if(F!=NULL) delete(F);
	F = new ForceExt_Constante(Vector3(0,-1,0),gravity->value());
	glWidget->getSystem()->getForcesExt()->setForce(F,0);
}
/****************************************************************************/	
void  WindowConfiguration_PCI_SPHSystem::changeMinS(double d)
{
	PCI_SPHSystem *system = (PCI_SPHSystem*) glWidget->getSystem();
	system->setMinS(Vector3(MinSX->value(),MinSY->value(),MinSZ->value()));
}
/****************************************************************************/	
void  WindowConfiguration_PCI_SPHSystem::changeMaxS(double d)
{
	PCI_SPHSystem *system = (PCI_SPHSystem*) glWidget->getSystem();
	system->setMaxS(Vector3(MaxSX->value(),MaxSY->value(),MaxSZ->value()));
}
/******************************************************************************************/
/******************************************************************************************/
