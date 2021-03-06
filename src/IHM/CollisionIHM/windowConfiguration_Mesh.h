#ifndef WINDOW_CONFIGURATION_MESH_H
#define WINDOW_CONFIGURATION_MESH_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
#include <ObjLoader.h>
//************************************************************************/
//************************************************************************/
class WindowConfiguration_Mesh : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Mesh(GLWidget *glWidget);
    ~WindowConfiguration_Mesh();

    MeshCollision *getMesh();

public slots:

    void accept();
    void cancel();
    void loadFile();
    void displayMesh(double);
    void changedIsContainer(int state);

private:
	
    MeshCollision *M;
    ObjLoader *loader;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;
    QPushButton *buttonLoad;

    QVBoxLayout *layout;
    QTabWidget *onglets;

    QDoubleSpinBox *elast, *friction;
    QCheckBox *container;

    QLabel *elastLabel, *frictionLabel;

    string getOpenFileName(const QString & caption, const QStringList & filters,int * ind_filter);
    QDir current_dir;
};

#endif
