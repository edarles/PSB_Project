#include <windowConfiguration_Box.h>

/****************************************************************************************************/
/****************************************************************************************************/
WindowConfiguration_Box::WindowConfiguration_Box(GLWidget *widget)
{
	onglets = new QTabWidget();
    	onglets->setGeometry(300, 200, 240, 160);

	QWidget *page1 = new QWidget();
	QWidget *page2 = new QWidget();
	onglets->addTab(page1,"Box");
	onglets->addTab(page2,"Parameters");

	widthLabel = new QLabel("width",page1);
 	width = new QDoubleSpinBox(page1);
	width->setMinimum(0.01);
	width->setValue(0.5);
	connect(width, SIGNAL(valueChanged(double)), this, SLOT(displayBox(double))); 
	lengthLabel = new QLabel("length",page1);
 	length = new QDoubleSpinBox(page1);
	length->setMinimum(0.01);
	length->setValue(0.5);
	connect(length, SIGNAL(valueChanged(double)), this, SLOT(displayBox(double))); 
	depthLabel = new QLabel("depth",page1);
 	depth = new QDoubleSpinBox(page1);
	depth->setValue(0.5);
	connect(depth, SIGNAL(valueChanged(double)), this, SLOT(displayBox(double))); 

	originLabel = new QLabel("Origin",page1);
	OX = new QDoubleSpinBox(page1);
	OY = new QDoubleSpinBox(page1);
	OZ = new QDoubleSpinBox(page1);
	OX->setMinimum(-10); OX->setValue(10.0);
	OY->setMinimum(-10); OY->setValue(10.0);
	OZ->setMinimum(-10); OZ->setValue(10.0);
	OX->setValue(0.0); OY->setValue(0.0); OZ->setValue(0.0);
	connect(OX, SIGNAL(valueChanged(double)), this, SLOT(displayBox(double))); 
	connect(OY, SIGNAL(valueChanged(double)), this, SLOT(displayBox(double))); 
	connect(OZ, SIGNAL(valueChanged(double)), this, SLOT(displayBox(double))); 

	QVBoxLayout *layout1 = new QVBoxLayout();
	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(originLabel,0,0);
	grid1->addWidget(OX,0,1); 
	grid1->addWidget(OY,0,2);
	grid1->addWidget(OZ,0,3);
	layout1->addLayout(grid1);
	QGridLayout *grid2 = new QGridLayout();
	grid2->addWidget(lengthLabel,0,0);
	grid2->addWidget(length,0,1);
	layout1->addLayout(grid2);
	QGridLayout *grid3 = new QGridLayout();
	grid3->addWidget(widthLabel,0,0);
	grid3->addWidget(width,0,1);
	layout1->addLayout(grid3);
	QGridLayout *grid4 = new QGridLayout();
	grid4->addWidget(depthLabel,0,0);
	grid4->addWidget(depth,0,1);
	layout1->addLayout(grid4);
	page1->setLayout(layout1);
	
	elastLabel = new QLabel("Restitution",page2);
	elast = new QDoubleSpinBox(page2);
	elast->setValue(0.45);
	elast->setMinimum(0.0);
	elast->setMaximum(1.0);
	connect(elast, SIGNAL(valueChanged(double)), this, SLOT(displayBox(double))); 
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
	B = new BoxCollision();
	B->create(Vector3(OX->value(),OY->value(),OZ->value()),length->value(),width->value(),depth->value(),elast->value(),container->isChecked());
	this->glWidget = widget;
	this->glWidget->getDisplay()->displayObjectCollision(B);
}
/****************************************************************************************************/
WindowConfiguration_Box::~WindowConfiguration_Box()
{
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_Box::accept()
{
	System *system = glWidget->getSystem();
	system->addObjectCollision(B);
	glWidget->getDisplay()->setObjectIsCreate();
	close();
}
/****************************************************************************************************/
void WindowConfiguration_Box::cancel()
{
	close();
}
/****************************************************************************************************/
/****************************************************************************************************/
BoxCollision* WindowConfiguration_Box::getBox()
{
	return B;
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_Box::displayBox(double d)
{
	if(B==NULL) delete(B);
	B = new BoxCollision();
	B->create(Vector3(OX->value(),OY->value(),OZ->value()),length->value(),width->value(),depth->value(),elast->value(),container->isChecked());
	glWidget->getDisplay()->displayObjectCollision(B);
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_Box::changedIsContainer(int state)
{
	if(state==0 && B!=NULL){
		if(B->getIsContainer()){
			B->setIsContainer(false);
			B->inverseNormales();
		}
	}
  	if(state==1 && B!=NULL){
		if(!B->getIsContainer()){
			B->setIsContainer(true);
			B->inverseNormales();
		}
	}
}
/****************************************************************************************************/
/****************************************************************************************************/
