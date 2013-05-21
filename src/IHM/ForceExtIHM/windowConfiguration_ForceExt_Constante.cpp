#include <windowConfiguration_ForceExt_Constante.h>

WindowConfiguration_ForceExt_Constante::WindowConfiguration_ForceExt_Constante(GLWidget *widget)
{
	DLabel = new QLabel("Direction",this);

	DX = new QDoubleSpinBox();
	DX->setValue(0);
	DX->setMinimum(-1.0); DX->setMaximum(1.0);
	connect(DX, SIGNAL(valueChanged(double)), this, SLOT(createForce(double))); 

	DY = new QDoubleSpinBox();
	DY->setValue(0);
	DY->setMinimum(-1.0); DY->setMaximum(1.0);
	connect(DY, SIGNAL(valueChanged(double)), this, SLOT(createForce(double))); 

	DZ = new QDoubleSpinBox();
	DZ->setValue(0);
	DZ->setMinimum(-1.0); DZ->setMaximum(1.0);
	connect(DZ, SIGNAL(valueChanged(double)), this, SLOT(createForce(double))); 

	ALabel = new QLabel("Amplitude",this);
	A = new QDoubleSpinBox();
	A->setValue(1.0);
	A->setMinimum(0.0);
	connect(A, SIGNAL(valueChanged(double)), this, SLOT(createForce(double))); 

	QVBoxLayout *layout = new QVBoxLayout();

	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(DLabel,0,0);
	grid1->addWidget(DX,0,1); 
	grid1->addWidget(DY,0,2); 
	grid1->addWidget(DZ,0,3); 
	layout->addLayout(grid1);

	QGridLayout *grid2 = new QGridLayout();
	grid2->addWidget(ALabel,0,0);
	grid2->addWidget(A,0,1);
	layout->addLayout(grid2);

	QGridLayout *grid6 = new QGridLayout();
	buttonOK = new QPushButton(tr("OK"),this);
	connect(buttonOK, SIGNAL(clicked()), this , SLOT(accept()));
	buttonCancel = new QPushButton(tr("Cancel"),this);
        connect(buttonCancel, SIGNAL(clicked()), this , SLOT(cancel()));
	grid6->addWidget(buttonOK,0,0);
	grid6->addWidget(buttonCancel,0,1);
	layout->addLayout(grid6);

	setLayout(layout);
	setWindowTitle("Configuration");

	this->glWidget = widget;
        F = new ForceExt_Constante(Vector3(DX->value(),DY->value(),DZ->value()),A->value());
}
WindowConfiguration_ForceExt_Constante::~WindowConfiguration_ForceExt_Constante()
{
}
QSize WindowConfiguration_ForceExt_Constante::sizeHint() const
{
     return QSize(300, 100);
}
void WindowConfiguration_ForceExt_Constante::accept()
{
	if(F!=NULL){
		glWidget->getSystem()->addForce(F);
	} 
	close();
}
void WindowConfiguration_ForceExt_Constante::cancel()
{
	close();
}	
void WindowConfiguration_ForceExt_Constante::createForce(double d)
{
	if(F!=NULL) delete(F);
	F = new ForceExt_Constante(Vector3(DX->value(),DY->value(),DZ->value()),A->value());
}
