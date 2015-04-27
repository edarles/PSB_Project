#include <windowConfiguration_LinearHeightField.h>
#include <windowConfiguration_CombinedHeightField.h>

/****************************************************************************************************/
/****************************************************************************************************/
WindowConfiguration_LinearHeightField::WindowConfiguration_LinearHeightField(WindowConfiguration_CombinedHeightField* parent, GLWidget *widget)
{
   this->parent = parent;

   if(parent==NULL){

	onglets = new QTabWidget();
    	onglets->setGeometry(300, 200, 240, 160);

	QWidget *page1 = new QWidget();
	QWidget *page2 = new QWidget();
	QWidget *page3 = new QWidget();
	onglets->addTab(page1,"HeightField");
	onglets->addTab(page2,"Linear");
	onglets->addTab(page3,"Parameters");
	
	originLabel = new QLabel("Origin",page1);
	OX = new QDoubleSpinBox(page1);
	OY = new QDoubleSpinBox(page1);
	OZ = new QDoubleSpinBox(page1);
	OX->setMinimum(-1000); OX->setMaximum(1000);
	OY->setMinimum(-1000); OY->setMaximum(1000);
	OZ->setMinimum(-1000); OZ->setMaximum(1000);
	OX->setValue(0.0); OY->setValue(-0.5); OZ->setValue(0.0);
	connect(OX, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 
	connect(OY, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 
	connect(OZ, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

	widthLabel = new QLabel("width",page1);
 	width = new QDoubleSpinBox(page1);
	width->setMinimum(0.01);
	width->setValue(1.0);
	connect(width, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

	lengthLabel = new QLabel("length",page1);
 	length = new QDoubleSpinBox(page1);
	length->setMinimum(0.01);
	length->setValue(1.0);
	connect(length, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

	dxLabel = new QLabel("dx",page1);
	dx = new QDoubleSpinBox(page1);
	dx->setMinimum(0.01);
	dx->setValue(0.05);
	connect(dx, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

	dzLabel = new QLabel("dz",page1);
	dz = new QDoubleSpinBox(page1);
	dz->setMinimum(0.01);
	dz->setValue(0.05);
	connect(dz, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

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
	grid4->addWidget(dxLabel,0,0);
	grid4->addWidget(dx,0,1);
	layout1->addLayout(grid4);

	QGridLayout *grid5 = new QGridLayout();
	grid5->addWidget(dzLabel,0,0);
	grid5->addWidget(dz,0,1);
	layout1->addLayout(grid5);

	page1->setLayout(layout1);

	aLabel = new QLabel("a",page2);
	a = new QDoubleSpinBox(page2);
	a->setValue(0.1);
	a->setMinimum(0.0);
	a->setMaximum(1.0);
	connect(a, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

	bLabel = new QLabel("b",page2);
	b = new QDoubleSpinBox(page2);
	b->setValue(0.1);
	b->setMinimum(0.0);
	b->setMaximum(1.0);
	connect(b, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

	QVBoxLayout *layout2 = new QVBoxLayout();

	QGridLayout *grid6 = new QGridLayout();
	grid6->addWidget(aLabel,0,0);
	grid6->addWidget(a,0,1);
	layout2->addLayout(grid6);

	QGridLayout *grid7 = new QGridLayout();
	grid7->addWidget(bLabel,0,0);
	grid7->addWidget(b,0,1);
	layout2->addLayout(grid7);

	page2->setLayout(layout2);

	elastLabel = new QLabel("Restitution",page3);
	elast = new QDoubleSpinBox(page3);
	elast->setValue(0.45);
	elast->setMinimum(0.0);
	elast->setMaximum(1.0);
	connect(elast, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 
	
	QVBoxLayout *layout3 = new QVBoxLayout();

	QGridLayout *grid9 = new QGridLayout();
	grid9->addWidget(elastLabel,0,0);
	grid9->addWidget(elast,0,1);
	layout3->addLayout(grid9);

	page3->setLayout(layout3);

	QGridLayout *grid10 = new QGridLayout();
	buttonOK = new QPushButton(tr("OK"),this);
	connect(buttonOK, SIGNAL(clicked()), this , SLOT(accept()));
	buttonCancel = new QPushButton(tr("Cancel"),this);
        connect(buttonCancel, SIGNAL(clicked()), this , SLOT(cancel()));
	grid10->addWidget(buttonOK,0,0);
	grid10->addWidget(buttonCancel,0,1);

	layout = new QVBoxLayout();
	layout->addWidget(onglets);
	layout->addLayout(grid10);

	setLayout(layout);
	setWindowTitle("Configuration");

	H = new Linear_HeightFieldCollision();
	H->create(Vector3(OX->value(),OY->value(),OZ->value()),(float)length->value(),(float)width->value(),dx->value(),dz->value(),a->value(),b->value(),elast->value());

	this->glWidget = widget;
	this->glWidget->getDisplay()->displayObjectCollision(H);

   }

   else {
 
	onglets = new QTabWidget();
    	onglets->setGeometry(300, 200, 240, 160);

	QWidget *page2 = new QWidget();
	onglets->addTab(page2,"Linear");
	
	aLabel = new QLabel("a",page2);
	a = new QDoubleSpinBox(page2);
	a->setValue(0.1);
	a->setMinimum(0.0);
	a->setMaximum(1.0);
	connect(a, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

	bLabel = new QLabel("b",page2);
	b = new QDoubleSpinBox(page2);
	b->setValue(0.1);
	b->setMinimum(0.0);
	b->setMaximum(1.0);
	connect(b, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

	QVBoxLayout *layout2 = new QVBoxLayout();

	QGridLayout *grid6 = new QGridLayout();
	grid6->addWidget(aLabel,0,0);
	grid6->addWidget(a,0,1);
	layout2->addLayout(grid6);

	QGridLayout *grid7 = new QGridLayout();
	grid7->addWidget(bLabel,0,0);
	grid7->addWidget(b,0,1);
	layout2->addLayout(grid7);

	page2->setLayout(layout2);

	QGridLayout *grid10 = new QGridLayout();
	buttonOK = new QPushButton(tr("Add"),this);
	connect(buttonOK, SIGNAL(clicked()), this , SLOT(add()));

	buttonCancel = new QPushButton(tr("Cancel"),this);
        connect(buttonCancel, SIGNAL(clicked()), this , SLOT(cancel()));
	grid10->addWidget(buttonOK,0,0);
	grid10->addWidget(buttonCancel,0,1);

	layout = new QVBoxLayout();
	layout->addWidget(onglets);
	layout->addLayout(grid10);

	setLayout(layout);
	setWindowTitle("Configuration");

	H = new Linear_HeightFieldCollision();
	H->create(Vector3(parent->OX->value(),parent->OY->value(),parent->OZ->value()),(float)parent->length->value(),(float)parent->width->value(),parent->dx->value(),parent->dz->value(),a->value(),b->value(),parent->elast->value());
	this->glWidget = widget;
	//this->glWidget->getDisplay()->displayObjectCollision(H);
   }
}
/****************************************************************************************************/
WindowConfiguration_LinearHeightField::~WindowConfiguration_LinearHeightField()
{
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_LinearHeightField::accept()
{
	System *system = glWidget->getSystem();
	system->addObjectCollision(H);
	glWidget->getDisplay()->setObjectIsCreate();
	close();
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_LinearHeightField::add()
{
	if(parent!=NULL){
		H = new Linear_HeightFieldCollision();
		H->create(Vector3(parent->OX->value(),parent->OY->value(),parent->OZ->value()),(float)parent->length->value(),(float)parent->width->value(),parent->dx->value(),parent->dz->value(),a->value(),b->value(),parent->elast->value());
		parent->Hfields.push_back(H);
		close();
	}
}
/****************************************************************************************************/
void WindowConfiguration_LinearHeightField::cancel()
{
	close();
}
/****************************************************************************************************/
/****************************************************************************************************/
Linear_HeightFieldCollision* WindowConfiguration_LinearHeightField::getHeightField()
{
	return H;
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_LinearHeightField::displayHeightField(double d)
{
	if(a->value()+b->value()>1.0){
		QMessageBox Q;
		Q.setIcon(QMessageBox::Critical);
		Q.setText("La somme des coefficients a et b doit être inférieure à 1 !!");
       	 	Q.exec();
		a->setValue(0.1);
		b->setValue(0.1);
	}
	else {
		if(H==NULL) delete(H);
		H = new Linear_HeightFieldCollision();
		if(parent==NULL)
			H->create(Vector3(OX->value(),OY->value(),OZ->value()),(float)length->value(),(float)width->value(),dx->value(),dz->value(),a->value(),b->value(),elast->value());
		else
			H->create(Vector3(parent->OX->value(),parent->OY->value(),parent->OZ->value()),(float)parent->length->value(),(float)parent->width->value(),parent->dx->value(),parent->dz->value(),a->value(),b->value(),parent->elast->value());
		glWidget->getDisplay()->displayObjectCollision(H);
	}
}
/****************************************************************************************************/
/****************************************************************************************************/
