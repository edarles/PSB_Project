#ifndef WINDOW_CONFIGURATION_EMITTER_MESH_H
#define WINDOW_CONFIGURATION_EMITTER_MESH_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
#include <ObjLoader.h>
//************************************************************************/
//************************************************************************/
class WindowConfiguration_Emitter_Mesh : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Emitter_Mesh(GLWidget *glWidget);
    ~WindowConfiguration_Emitter_Mesh();

    EmitterMesh *getEmitter();

public slots:

    void accept();
    void cancel();
    void loadFile();
    void displayMesh(double);
    void displayMesh(int);

private:
	
    EmitterMesh *B;
    Mesh mesh;
    GLWidget *glWidget;
    
    QPushButton *buttonLoad;
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QVBoxLayout *layout;

    QDoubleSpinBox *VX, *VY, *VZ;
    QSpinBox *maxEmission, *durationTime;
    QLabel *velocityLabel,*maxEmissionLabel, *durationTimeLabel;

    string getOpenFileName(const QString & caption, const QStringList & filters,int * ind_filter);
    QDir current_dir;
    
};

#endif
