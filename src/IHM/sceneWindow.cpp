#include <sceneWindow.h>
#include <typeinfo>

/*************************************************************************/
/*************************************************************************/
SceneWindow::SceneWindow():QScrollArea()
{
	setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	setStyleSheet(QString::fromUtf8("background-color: rgb(200, 200, 200);"));
	//setAlignment(Qt::AlignHCenter);  
	tabWidget = new QTabWidget(this);
	tabWidget->setStyleSheet(QString::fromUtf8("background-color: rgb(200, 200, 200);"));
/*
	mainLayout->addWidget(tabWidget);
	setLayout(mainLayout);
*/ 
	QFrame *container = new QFrame();  
	mainLayout = new QVBoxLayout(container);
	QWidget *page = createPage_SimpleSystem();
	tabWidget->addTab(page,"System");
	//tabWidget->setTabTextColor(0,Qt::blue);

        mainLayout->addWidget(tabWidget);
        container->setLayout(mainLayout);
	this->setWidget(container);
}
/*************************************************************************/
/*************************************************************************/
void SceneWindow::clearTabWidget()
{
	tabWidget->clear();
}
/*************************************************************************/
/*************************************************************************/
void SceneWindow::addPageSystem()
{
 //  if(typeid(*S) == typeid(SimpleSystem))
 //	QWidget *page = createPage_SimpleSystem();
 /*  tabWidget->addTab(page,"System");
   tabWidget->show();
   mainLayout->addWidget(tabWidget);
   setLayout(mainLayout);*/
}
/*************************************************************************/
/*void SceneWindow::addPageEmitterSystem(System*)
{
}
/*************************************************************************/
/*void SceneWindow::addPageCollisionSystem(System*)
{
}
/*************************************************************************/
QWidget* SceneWindow::createPage_SimpleSystem()
{
	QWidget *page = new QWidget();
	QLabel *vide = new QLabel("                                    ",page);

	QLabel *gravityLabel = new QLabel("Gravity",page);
	gravityLabel->setStyleSheet("QLabel {  color : black; }");
	QLabel* gravity = new QLabel("9.81",page);
	gravity->setStyleSheet("QLabel { background-color : black; color : white; }");
	gravity->setAlignment(Qt::AlignHCenter);  

	QLabel* particleRadiusLabel = new QLabel("radius",page);
	particleRadiusLabel->setStyleSheet("QLabel {  color : black; }");
 	QLabel* particleRadius = new QLabel("0.02",page);
	particleRadius->setStyleSheet("QLabel { background-color : black; color : white; }");
	particleRadius->setAlignment(Qt::AlignHCenter);  

	QLabel* particleMassLabel = new QLabel("Mass",page);
	particleMassLabel->setStyleSheet("QLabel {  color : black; }");
	QLabel* particleMass = new QLabel("0.02",page);
	particleMass->setStyleSheet("QLabel { background-color : black; color : white; }");
	particleMass->setAlignment(Qt::AlignHCenter);  

	QLabel* deltaTimeLabel = new QLabel("Time step",page);
	deltaTimeLabel->setStyleSheet("QLabel {  color : black; }");
 	QLabel* deltaTime = new QLabel("0.01",page);
	deltaTime->setStyleSheet("QLabel { background-color : black; color : white; }");
	deltaTime->setAlignment(Qt::AlignHCenter);  

	QVBoxLayout *layout = new QVBoxLayout();

	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(vide,0,0); 
	grid1->addWidget(vide,0,1); 
	grid1->addWidget(gravityLabel,1,0);
	grid1->addWidget(gravity,1,1); 
	grid1->addWidget(vide,2,0); 
	grid1->addWidget(vide,2,1); 
	layout->addLayout(grid1);

	QGridLayout *grid2 = new QGridLayout();
	grid2->addWidget(particleRadiusLabel,0,0);
	grid2->addWidget(particleRadius,0,1);
	grid2->addWidget(vide,1,0); 
	grid2->addWidget(vide,1,1); 
	layout->addLayout(grid2);

	QGridLayout *grid5 = new QGridLayout();
	grid5->addWidget(particleMassLabel,0,0);
	grid5->addWidget(particleMass,0,1);
	grid5->addWidget(vide,1,0); 
	grid5->addWidget(vide,1,1); 
	layout->addLayout(grid5);

	QGridLayout *grid3 = new QGridLayout();
	grid3->addWidget(deltaTimeLabel,0,0);
	grid3->addWidget(deltaTime,0,1);
	grid3->addWidget(vide,1,0); 
	grid3->addWidget(vide,1,1); 
	layout->addLayout(grid3);

	/*QLabel *imageLabel = new QLabel;
	QImage image("icons/simpleSystem.png");
	imageLabel->setPixmap(QPixmap::fromImage(image));
	layout->addWidget(imageLabel);
*/
	page->setLayout(layout);
	return page;
}
/*************************************************************************/
/*************************************************************************/
