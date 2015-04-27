/**********************************************************************************/
/**********************************************************************************/
#include <windowConfiguration_Plan.h>

/**********************************************************************************/
WindowConfiguration_Plan::WindowConfiguration_Plan(GLWidget *widget)
{
	onglets = new QTabWidget();
    	onglets->setGeometry(300, 200, 240, 160);

	QWidget *page1 = new QWidget();
	QWidget *page2 = new QWidget();
	onglets->addTab(page1,"Plan");
	onglets->addTab(page2,"Parameters");

	widthLabel = new QLabel("width",page1);
 	width = new QDoubleSpinBox(page1);
	width->setValue(3.0);
	connect(width, SIGNAL(valueChanged(double)), this, SLOT(displayPlan(double))); 
	lengthLabel = new QLabel("length",page1);
 	length = new QDoubleSpinBox(page1);
	length->setValue(3.0);
	connect(length, SIGNAL(valueChanged(double)), this, SLOT(displayPlan(double))); 

	originLabel = new QLabel("Origin",page1);
	OX = new QDoubleSpinBox(page1);
	OY = new QDoubleSpinBox(page1);
	OZ = new QDoubleSpinBox(page1);
	OX->setValue(0.0); OY->setValue(0.0); OZ->setValue(0.0);
	OX->setMinimum(-10000); OX->setMaximum(10000);
	OY->setMinimum(-10000); OY->setMaximum(10000);
	OZ->setMinimum(-10000); OZ->setMaximum(10000);
	connect(OX, SIGNAL(valueChanged(double)), this, SLOT(displayPlan(double))); 
	connect(OY, SIGNAL(valueChanged(double)), this, SLOT(displayPlan(double))); 
	connect(OZ, SIGNAL(valueChanged(double)), this, SLOT(displayPlan(double))); 

	directionLabel = new QLabel("Normale",page1);
	DX = new QDoubleSpinBox(page1);
	DY = new QDoubleSpinBox(page1);
	DZ = new QDoubleSpinBox(page1);
	DX->setValue(0.0); DY->setValue(1.0); DZ->setValue(0.0);
	DX->setMinimum(-1.0); DX->setMaximum(1.0); 
	DY->setMinimum(-1.0); DY->setMaximum(1.0); 
	DZ->setMinimum(-1.0); DZ->setMaximum(1.0); 
	connect(DX, SIGNAL(valueChanged(double)), this, SLOT(displayPlan(double))); 
	connect(DY, SIGNAL(valueChanged(double)), this, SLOT(displayPlan(double))); 
	connect(DZ, SIGNAL(valueChanged(double)), this, SLOT(displayPlan(double))); 

	QVBoxLayout *layout1 = new QVBoxLayout();
	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(originLabel,0,0);
	grid1->addWidget(OX,0,1); 
	grid1->addWidget(OY,0,2);
	grid1->addWidget(OZ,0,3);
	layout1->addLayout(grid1);
        QGridLayout *grid8 = new QGridLayout();
	grid8->addWidget(directionLabel,0,0);
	grid8->addWidget(DX,0,1);
	grid8->addWidget(DY,0,2);
	grid8->addWidget(DZ,0,3);
	layout1->addLayout(grid8);
	QGridLayout *grid2 = new QGridLayout();
	grid2->addWidget(lengthLabel,0,0);
	grid2->addWidget(length,0,1);
	layout1->addLayout(grid2);
	QGridLayout *grid3 = new QGridLayout();
	grid3->addWidget(widthLabel,0,0);
	grid3->addWidget(width,0,1);
	layout1->addLayout(grid3);
	page1->setLayout(layout1);
	
	elastLabel = new QLabel("Restitution",page2);
	elast = new QDoubleSpinBox(page2);
	elast->setValue(0.8);
	elast->setMinimum(0.0);
	elast->setMaximum(1.0);
	connect(elast, SIGNAL(valueChanged(double)), this, SLOT(displayPlan(double))); 

	QVBoxLayout *layout2 = new QVBoxLayout();
	QGridLayout *grid5 = new QGridLayout();
	grid5->addWidget(elastLabel,0,0);
	grid5->addWidget(elast,0,1);
	layout2->addLayout(grid5);
	
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
	B = new PlaneCollision();
	B->create(Vector3(OX->value(),OY->value(),OZ->value()),length->value(),width->value(),elast->value(),1-elast->value(),false,Vector3(DX->value(),DY->value(),DZ->value()));
	this->glWidget = widget;
}
/**********************************************************************************/
WindowConfiguration_Plan::~WindowConfiguration_Plan()
{
}
/**********************************************************************************/
void WindowConfiguration_Plan::accept()
{
	System *system = glWidget->getSystem();
	system->addObjectCollision(B);
	close();
}
/**********************************************************************************/
void WindowConfiguration_Plan::cancel()
{
	close();
}
/**********************************************************************************/
PlaneCollision* WindowConfiguration_Plan::getPlan()
{
	return B;
}
/**********************************************************************************/
void WindowConfiguration_Plan::displayPlan(double d)
{
	if(B==NULL) delete(B);
	B = new PlaneCollision();
	B->create(Vector3(OX->value(),OY->value(),OZ->value()),length->value(),width->value(),elast->value(),1-elast->value(),false,Vector3(DX->value(),DY->value(),DZ->value()));

}
/**********************************************************************************/
/**********************************************************************************/
