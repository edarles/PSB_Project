#include <windowConfiguration_Cylinder.h>

/****************************************************************************************************/
/****************************************************************************************************/
WindowConfiguration_Cylinder::WindowConfiguration_Cylinder(GLWidget *widget)
{
	onglets = new QTabWidget();
    	onglets->setGeometry(300, 200, 240, 160);

	QWidget *page1 = new QWidget();
	QWidget *page2 = new QWidget();
	onglets->addTab(page1,"Cylinder");
	onglets->addTab(page2,"Parameters");

	originLabel = new QLabel("Center",page1);
	OX = new QDoubleSpinBox(page1);
	OY = new QDoubleSpinBox(page1);
	OZ = new QDoubleSpinBox(page1);
	OX->setValue(0.0); OY->setValue(0.0); OZ->setValue(0.0);
	connect(OX, SIGNAL(valueChanged(double)), this, SLOT(displayCylinder(double))); 
	connect(OY, SIGNAL(valueChanged(double)), this, SLOT(displayCylinder(double))); 
	connect(OZ, SIGNAL(valueChanged(double)), this, SLOT(displayCylinder(double))); 

	lengthLabel = new QLabel("Length",page1);
 	length = new QDoubleSpinBox(page1);
	length->setMinimum(0.01);
	length->setMaximum(1000);
	length->setValue(1.0);
	connect(length, SIGNAL(valueChanged(double)), this, SLOT(displayCylinder(double))); 

	radiusLabel = new QLabel("Radius",page1);
 	radius = new QDoubleSpinBox(page1);
	radius->setMinimum(0.01);
	radius->setMaximum(1000);
	radius->setValue(0.5);
	connect(radius, SIGNAL(valueChanged(double)), this, SLOT(displayCylinder(double))); 

	directionLabel = new QLabel("Direction",page1);
	DX = new QDoubleSpinBox(page1);
	DY = new QDoubleSpinBox(page1);
	DZ = new QDoubleSpinBox(page1);
	DX->setMinimum(-1.0); DX->setMaximum(1.0);
	DY->setMinimum(-1.0); DY->setMaximum(1.0);
	DZ->setMinimum(-1.0); DZ->setMaximum(1.0);
	DX->setValue(0.0); DY->setValue(1.0); DZ->setValue(0.0);
	connect(DX, SIGNAL(valueChanged(double)), this, SLOT(displayCylinder(double))); 
	connect(DY, SIGNAL(valueChanged(double)), this, SLOT(displayCylinder(double))); 
	connect(DZ, SIGNAL(valueChanged(double)), this, SLOT(displayCylinder(double))); 

	QVBoxLayout *layout1 = new QVBoxLayout();

	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(originLabel,0,0);
	grid1->addWidget(OX,0,1); 
	grid1->addWidget(OY,0,2);
	grid1->addWidget(OZ,0,3);
	layout1->addLayout(grid1);

	QGridLayout *grid2 = new QGridLayout();
	grid2->addWidget(directionLabel,0,0);
	grid2->addWidget(DX,0,1); 
	grid2->addWidget(DY,0,2);
	grid2->addWidget(DZ,0,3);
	layout1->addLayout(grid2);

	QGridLayout *grid3 = new QGridLayout();
	grid3->addWidget(lengthLabel,0,0);
	grid3->addWidget(length,0,1); 
	layout1->addLayout(grid3);

	QGridLayout *grid4 = new QGridLayout();
	grid4->addWidget(radiusLabel,0,0);
	grid4->addWidget(radius,0,1); 
	layout1->addLayout(grid4);

	page1->setLayout(layout1);
	
	elastLabel = new QLabel("Restitution",page2);
	elast = new QDoubleSpinBox(page2);
	elast->setValue(0.45);
	elast->setMinimum(0.0);
	elast->setMaximum(1.0);
	connect(elast, SIGNAL(valueChanged(double)), this, SLOT(displayCylinder(double))); 
	container = new QCheckBox(tr("container"),page2);
	container->setChecked(true);
	connect(container,SIGNAL(stateChanged(int)), this, SLOT(changedContainer(int)));
 
	QVBoxLayout *layout2 = new QVBoxLayout();
	QGridLayout *grid5 = new QGridLayout();
	grid5->addWidget(elastLabel,0,0);
	grid5->addWidget(elast,0,1);
	layout2->addLayout(grid5);
	layout2->addWidget(container);
	page2->setLayout(layout2);

	QGridLayout *grid7 = new QGridLayout();
	buttonOK = new QPushButton(tr("OK"),this);
	connect(buttonOK, SIGNAL(clicked()), this , SLOT(accept()));
	buttonCancel = new QPushButton(tr("Cancel"),this);
        connect(buttonCancel, SIGNAL(clicked()), this , SLOT(cancel()));
	grid7->addWidget(buttonOK,0,0);
	grid7->addWidget(buttonCancel,0,1);

	layout = new QVBoxLayout();
	layout->addWidget(onglets);
	layout->addLayout(grid7);

	setLayout(layout);
	setWindowTitle("Configuration");
	C = new CylinderCollision();

	C->create(Vector3(OX->value(),OY->value(),OZ->value()),radius->value(),length->value(),
		  Vector3(DX->value(),DY->value(),DZ->value()),elast->value(),container->isChecked());
	this->glWidget = widget;
	//this->glWidget->getDisplay()->displayObjectCollision(B);
}
/****************************************************************************************************/
WindowConfiguration_Cylinder::~WindowConfiguration_Cylinder()
{
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_Cylinder::accept()
{
	System *system = glWidget->getSystem();
	system->addObjectCollision(C);
	glWidget->getDisplay()->setObjectIsCreate();
	close();
}
/****************************************************************************************************/
void WindowConfiguration_Cylinder::cancel()
{
	close();
}
/****************************************************************************************************/
/****************************************************************************************************/
CylinderCollision* WindowConfiguration_Cylinder::getCylinder()
{
	return C;
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_Cylinder::displayCylinder(double d)
{
	if(C==NULL) delete(C);
	C = new CylinderCollision();
	
	C->create(Vector3(OX->value(),OY->value(),OZ->value()),radius->value(),length->value(),
		  Vector3(DX->value(),DY->value(),DZ->value()),elast->value(),container->isChecked());
	glWidget->getDisplay()->displayObjectCollision(C);
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_Cylinder::changedIsContainer(int state)
{
	if(state==0 && C!=NULL){
		if(C->getIsContainer()){
			C->setIsContainer(false);
			C->inverseNormales();
		}
	}
  	if(state==1 && C!=NULL){
		if(!C->getIsContainer()){
			C->setIsContainer(true);
			C->inverseNormales();
		}
	}
}
/****************************************************************************************************/
/****************************************************************************************************/
