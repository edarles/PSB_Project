#include <windowConfiguration_Sphere.h>

WindowConfiguration_Sphere::WindowConfiguration_Sphere(GLWidget *widget)
{
	onglets = new QTabWidget();
    	onglets->setGeometry(300, 200, 240, 160);

	QWidget *page1 = new QWidget();
	QWidget *page2 = new QWidget();
	onglets->addTab(page1,"Sphere");
	onglets->addTab(page2,"Parameters");

	radiusLabel = new QLabel("Radius",page1);
 	radius = new QDoubleSpinBox(page1);
	radius->setValue(1.0);
	connect(radius, SIGNAL(valueChanged(double)), this, SLOT(displaySphere(double))); 

	originLabel = new QLabel("Origin",page1);
	OX = new QDoubleSpinBox(page1);
	OY = new QDoubleSpinBox(page1);
	OZ = new QDoubleSpinBox(page1);
	OX->setValue(0.0); OY->setValue(0.0); OZ->setValue(0.0);
	connect(OX, SIGNAL(valueChanged(double)), this, SLOT(displaySphere(double))); 
	connect(OY, SIGNAL(valueChanged(double)), this, SLOT(displaySphere(double))); 
	connect(OZ, SIGNAL(valueChanged(double)), this, SLOT(displaySphere(double))); 

	QVBoxLayout *layout1 = new QVBoxLayout();
	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(originLabel,0,0);
	grid1->addWidget(OX,0,1); 
	grid1->addWidget(OY,0,2);
	grid1->addWidget(OZ,0,3);
	layout1->addLayout(grid1);
	QGridLayout *grid2 = new QGridLayout();
	grid2->addWidget(radiusLabel,0,0);
	grid2->addWidget(radius,0,1);
	layout1->addLayout(grid2);
	page1->setLayout(layout1);

	elastLabel = new QLabel("Damping",page2);
	elast = new QDoubleSpinBox(page2);
	elast->setValue(0.8);
	elast->setMinimum(0.0);
	elast->setMaximum(1.0);
	frictionLabel = new QLabel("Friction",page2);
	friction = new QDoubleSpinBox(page2);
	friction->setValue(0.2);
	friction->setMinimum(0.0);
	friction->setMaximum(1.0);
	container = new QCheckBox(tr("container"),page2);
        container->setChecked(true);
	connect(container,SIGNAL(stateChanged(int)), this, SLOT(changedContainer(int)));

	QVBoxLayout *layout2 = new QVBoxLayout();
	QGridLayout *grid5 = new QGridLayout();
	grid5->addWidget(elastLabel,0,0);
	grid5->addWidget(elast,0,1);
	layout2->addLayout(grid5);
	QGridLayout *grid6 = new QGridLayout();
	grid6->addWidget(frictionLabel,0,0);
	grid6->addWidget(friction,0,1);
	layout2->addLayout(grid6);
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
	S = new SphereCollision(Vector3(OX->value(),OY->value(),OZ->value()),radius->value(),elast->value(),friction->value(),container->isChecked());
	this->glWidget = widget;
	this->glWidget->getDisplay()->displayObjectCollision(S);
}
WindowConfiguration_Sphere::~WindowConfiguration_Sphere()
{
}
void WindowConfiguration_Sphere::accept()
{
	System *system = glWidget->getSystem();
	system->addObjectCollision(S);
	glWidget->getDisplay()->setObjectIsCreate();
	close();
}
void WindowConfiguration_Sphere::cancel()
{
	close();
}

SphereCollision* WindowConfiguration_Sphere::getSphere()
{
	return S;
}
void WindowConfiguration_Sphere::displaySphere(double d)
{
	if(S==NULL) delete(S);
	S = new SphereCollision(Vector3(OX->value(),OY->value(),OZ->value()),radius->value(),elast->value(),friction->value(),container->isChecked());
	glWidget->getDisplay()->displayObjectCollision(S);
}
void WindowConfiguration_Sphere::changedIsContainer(int state)
{
	if(state==0 && S!=NULL){
		if(S->getIsContainer()){
			S->setIsContainer(false);
		}
	}
	if(state==1 && S!=NULL){
		if(S->getIsContainer()){
			S->setIsContainer(true);
		}
	}
}
