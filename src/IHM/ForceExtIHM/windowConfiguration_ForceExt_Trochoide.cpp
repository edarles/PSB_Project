#include <windowConfiguration_ForceExt_Trochoide.h>

WindowConfiguration_ForceExt_Trochoide::WindowConfiguration_ForceExt_Trochoide(GLWidget *widget)
{
	ALabel = new QLabel("Amplitude",this);
	A = new QDoubleSpinBox();
	A->setValue(0.5);
	connect(A, SIGNAL(valueChanged(double)), this, SLOT(createForce(double))); 
	
	lambdaLabel = new QLabel("Longueur d'onde",this);
	lambda = new QDoubleSpinBox();
	lambda->setValue(2.80);
	connect(lambda, SIGNAL(valueChanged(double)), this, SLOT(createForce(double))); 

	thetaLabel = new QLabel("Deviation angle",this);
	theta = new QDoubleSpinBox();
	theta->setValue(7.8);
	connect(theta, SIGNAL(valueChanged(double)), this, SLOT(createForce(double))); 

	fLabel = new QLabel("Frequency",this);
	f = new QDoubleSpinBox();
	f->setValue(1.33);
	connect(f, SIGNAL(valueChanged(double)), this, SLOT(createForce(double))); 

	phiLabel = new QLabel("Dephasage",this);
	phi = new QDoubleSpinBox();
	phi->setValue(0);
	connect(phi, SIGNAL(valueChanged(double)), this, SLOT(createForce(double))); 

	QVBoxLayout *layout = new QVBoxLayout();

	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(ALabel,0,0);
	grid1->addWidget(A,0,1); 
	layout->addLayout(grid1);

	QGridLayout *grid2 = new QGridLayout();
	grid2->addWidget(lambdaLabel,0,0);
	grid2->addWidget(lambda,0,1);
	layout->addLayout(grid2);

	QGridLayout *grid3 = new QGridLayout();
	grid3->addWidget(thetaLabel,0,0);
	grid3->addWidget(theta,0,1);
	layout->addLayout(grid3);

	QGridLayout *grid4 = new QGridLayout();
	grid4->addWidget(fLabel,0,0);
	grid4->addWidget(f,0,1);
	layout->addLayout(grid4);

	QGridLayout *grid5 = new QGridLayout();
	grid5->addWidget(phiLabel,0,0);
	grid5->addWidget(phi,0,1);
	layout->addLayout(grid5);

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
        T = new ForceExt_Trochoide(A->value(),lambda->value(),theta->value(),f->value(),phi->value());
}
WindowConfiguration_ForceExt_Trochoide::~WindowConfiguration_ForceExt_Trochoide()
{
}
QSize WindowConfiguration_ForceExt_Trochoide::sizeHint() const
{
     return QSize(300, 100);
}
void WindowConfiguration_ForceExt_Trochoide::accept()
{
	if(T!=NULL){
		printf("A:%f\n",T->getAmplitude());
		glWidget->getSystem()->addForce(T);
	} 
	close();
}
void WindowConfiguration_ForceExt_Trochoide::cancel()
{
	close();
}	
void WindowConfiguration_ForceExt_Trochoide::createForce(double d)
{
	if(T!=NULL) delete(T);
	T = new ForceExt_Trochoide(A->value(),lambda->value(),theta->value(),f->value(),phi->value());
}
