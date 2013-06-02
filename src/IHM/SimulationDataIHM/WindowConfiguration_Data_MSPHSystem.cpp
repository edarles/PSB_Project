#include <WindowConfiguration_Data_MSPHSystem.h>

/*********************************************************************************************/
/*********************************************************************************************/
WindowConfiguration_Data_MSPHSystem::WindowConfiguration_Data_MSPHSystem(QWidget* widget):WindowConfiguration_Data(widget)
{
	particleRadiusLabel = new QLabel("Particle Radius",widget);
 	particleRadius = new QDoubleSpinBox(widget);
	particleRadius->setMinimum(0.01);
	particleRadius->setMaximum(1.0);
	particleRadius->setValue(0.02);
	connect(particleRadius, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	particleMassLabel = new QLabel("Particle Mass",widget);
 	particleMass = new QDoubleSpinBox(widget);
	particleMass->setMinimum(0.01);
	particleMass->setMaximum(100.0);
	particleMass->setValue(0.02);
	connect(particleMass, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	restDensityLabel = new QLabel("Rest Density",widget);
	restDensity = new QDoubleSpinBox(widget);
	restDensity->setMinimum(1.0);
	restDensity->setMaximum(10000);
	restDensity->setValue(998.29);
        connect(restDensity, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	viscosityLabel = new QLabel("Viscosity",widget);
	viscosity = new QDoubleSpinBox(widget);
	viscosity->setMinimum(0.0);
	viscosity->setMaximum(100);
	viscosity->setValue(3.5);
	connect(viscosity, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	surfaceTensionLabel = new QLabel("Surface tension",widget);
	surfaceTension = new QDoubleSpinBox(widget);
	surfaceTension->setDecimals(4);
	surfaceTension->setMinimum(0.0);
	surfaceTension->setMaximum(100);
	surfaceTension->setValue(0.0728);
	connect(surfaceTension, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	gasStiffnessLabel = new QLabel("Gas Stiffness",widget);
	gasStiffness = new QDoubleSpinBox(widget);
	gasStiffness->setMinimum(0.0);
	gasStiffness->setMaximum(100);
	gasStiffness->setValue(3);
	connect(gasStiffness, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	kernelParticlesLabel = new QLabel("Kernel Particles",widget);
	kernelParticles = new QDoubleSpinBox(widget);
	kernelParticles->setMinimum(0.0);
	kernelParticles->setMaximum(100);
	kernelParticles->setValue(20);
	connect(kernelParticles, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	temperatureLabel = new QLabel ("Temperature",widget);
	temperature = new QDoubleSpinBox(widget);
	temperature->setMinimum(0.0);
	temperature->setMaximum(200);
	temperature->setValue(20);
	connect(temperature, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	sigmaLabel = new QLabel ("sigma",widget);
	sigmaX = new QDoubleSpinBox(widget); sigmaY = new QDoubleSpinBox(widget); sigmaZ = new QDoubleSpinBox(widget);
	sigmaX->setMinimum(0.0); sigmaY->setMinimum(0.0); sigmaZ->setMinimum(0.0);
	sigmaX->setMaximum(200); sigmaY->setMaximum(200); sigmaZ->setMaximum(200);
	sigmaX->setValue(20); sigmaY->setValue(20); sigmaZ->setValue(20);
        connect(sigmaX, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 
	connect(sigmaY, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 
	connect(sigmaZ, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	betaLabel = new QLabel ("beta",widget);
	betaX = new QDoubleSpinBox(widget); betaY = new QDoubleSpinBox(widget); betaZ = new QDoubleSpinBox(widget);
	betaX->setMinimum(0.0); betaY->setMinimum(0.0); betaZ->setMinimum(0.0);
	betaX->setMaximum(200); betaY->setMaximum(200); betaZ->setMaximum(200);
	betaX->setValue(20); betaY->setValue(20); betaZ->setValue(20);
        connect(betaX, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 
	connect(betaY, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 
	connect(betaZ, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	gLabel = new QLabel ("g",widget);
	gX = new QDoubleSpinBox(widget); gY = new QDoubleSpinBox(widget); gZ = new QDoubleSpinBox(widget);
	gX->setMinimum(0.0); gY->setMinimum(0.0); gZ->setMinimum(0.0);
	gX->setMaximum(200); gY->setMaximum(200); gZ->setMaximum(200);
	gX->setValue(20); gY->setValue(20); gZ->setValue(20);
        connect(gX, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 
	connect(gY, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 
	connect(gZ, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	colorButton = new QPushButton("Particles color",this);
	connect(colorButton, SIGNAL(pressed()),this, SLOT(setColor()));

	QVBoxLayout *layout1 = new QVBoxLayout();

	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(particleRadiusLabel,0,0);
	grid1->addWidget(particleRadius,0,1); 
	layout1->addLayout(grid1);

	QGridLayout *grid2 = new QGridLayout();
	grid2->addWidget(particleMassLabel,0,0);
	grid2->addWidget(particleMass,0,1); 
	layout1->addLayout(grid2);

	QGridLayout *grid3 = new QGridLayout();
	grid3->addWidget(restDensityLabel,0,0);
	grid3->addWidget(restDensity,0,1); 
	layout1->addLayout(grid3);

	QGridLayout *grid4 = new QGridLayout();
	grid4->addWidget(viscosityLabel,0,0);
	grid4->addWidget(viscosity,0,1); 
	layout1->addLayout(grid4);

	QGridLayout *grid5 = new QGridLayout();
	grid5->addWidget(surfaceTensionLabel,0,0);
	grid5->addWidget(surfaceTension,0,1); 
	layout1->addLayout(grid5);

	QGridLayout *grid6 = new QGridLayout();
	grid6->addWidget(gasStiffnessLabel,0,0);
	grid6->addWidget(gasStiffness,0,1); 
	layout1->addLayout(grid6);

	QGridLayout *grid7 = new QGridLayout();
	grid7->addWidget(kernelParticlesLabel,0,0);
	grid7->addWidget(kernelParticles,0,1); 
	layout1->addLayout(grid7);

	QGridLayout *grid8 = new QGridLayout();
	grid8->addWidget(temperatureLabel,0,0);
	grid8->addWidget(temperature,0,1); 
	layout1->addLayout(grid8);

	QGridLayout *grid9 = new QGridLayout();
	grid9->addWidget(sigmaLabel,0,0);
	grid9->addWidget(sigmaX,0,1); 
	grid9->addWidget(sigmaY,0,2); 
	grid9->addWidget(sigmaZ,0,3); 
	layout1->addLayout(grid9);

	QGridLayout *grid10 = new QGridLayout();
	grid10->addWidget(betaLabel,0,0);
	grid10->addWidget(betaX,0,1); 
	grid10->addWidget(betaY,0,2); 
	grid10->addWidget(betaZ,0,3); 
	layout1->addLayout(grid10);

	QGridLayout *grid11 = new QGridLayout();
	grid11->addWidget(gLabel,0,0);
	grid11->addWidget(gX,0,1); 
	grid11->addWidget(gY,0,2); 
	grid11->addWidget(gZ,0,3); 
	layout1->addLayout(grid11);

	QGridLayout *grid12 = new QGridLayout();
	grid12->addWidget(colorButton,0,0); 
	layout1->addLayout(grid12);

	widget->setLayout(layout1);
    
        color = QColor(255,0,255);
	
	data = new SimulationData_MSPHSystem(particleRadius->value(),particleMass->value(),Vector3(color.red()/255,color.green()/255,
		   color.blue()/255),restDensity->value(),viscosity->value(),surfaceTension->value(),gasStiffness->value(),
		   kernelParticles->value(),temperature->value(),Vector3(sigmaX->value(),sigmaY->value(),sigmaZ->value()),
		   Vector3(betaX->value(),betaY->value(),betaZ->value()),Vector3(gX->value(),gY->value(),gZ->value()));
}
/*********************************************************************************************/
WindowConfiguration_Data_MSPHSystem::~WindowConfiguration_Data_MSPHSystem()
{
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Data_MSPHSystem::changeData(double newValue)
{
	if(data!=NULL)	delete(data);
	data = new SimulationData_MSPHSystem(particleRadius->value(),particleMass->value(),Vector3(color.red()/255,color.green()/255,
		   color.blue()/255),restDensity->value(),viscosity->value(),surfaceTension->value(),gasStiffness->value(),
		   kernelParticles->value(),temperature->value(),Vector3(sigmaX->value(),sigmaY->value(),sigmaZ->value()),
		   Vector3(betaX->value(),betaY->value(),betaZ->value()),Vector3(gX->value(),gY->value(),gZ->value()));
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Data_MSPHSystem::changeData(QColor newValue)
{
	if(data!=NULL)	delete(data);
	data = new SimulationData_MSPHSystem(particleRadius->value(),particleMass->value(),
	       Vector3((double)(newValue.red()/255),(double)(newValue.green()/255),(double)(newValue.blue()/255)),
	       restDensity->value(),viscosity->value(),surfaceTension->value(),gasStiffness->value(),kernelParticles->value(),
	       temperature->value(),Vector3(sigmaX->value(),sigmaY->value(),sigmaZ->value()),
	       Vector3(betaX->value(),betaY->value(),betaZ->value()),Vector3(gX->value(),gY->value(),gZ->value()));
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Data_MSPHSystem::setColor()
{
	QColorDialog *colorDialog = new QColorDialog();
	connect(colorDialog,SIGNAL(currentColorChanged(QColor)),this, SLOT(changeData(QColor)));
	colorDialog->open();
}
/*********************************************************************************************/
/*********************************************************************************************/