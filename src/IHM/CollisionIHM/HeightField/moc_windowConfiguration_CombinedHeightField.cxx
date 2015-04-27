/****************************************************************************
** Meta object code from reading C++ file 'windowConfiguration_CombinedHeightField.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "windowConfiguration_CombinedHeightField.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'windowConfiguration_CombinedHeightField.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_WindowConfiguration_CombinedHeightField[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      40,   49,   49,   49, 0x0a,
      50,   49,   49,   49, 0x0a,
      59,   49,   49,   49, 0x0a,
      86,   49,   49,   49, 0x0a,
     109,   49,   49,   49, 0x0a,
     134,   49,   49,   49, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_WindowConfiguration_CombinedHeightField[] = {
    "WindowConfiguration_CombinedHeightField\0"
    "accept()\0\0cancel()\0displayHeightField(double)\0"
    "addLinearHeightField()\0addGaussianHeightField()\0"
    "addPeriodicHeightField()\0"
};

void WindowConfiguration_CombinedHeightField::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        WindowConfiguration_CombinedHeightField *_t = static_cast<WindowConfiguration_CombinedHeightField *>(_o);
        switch (_id) {
        case 0: _t->accept(); break;
        case 1: _t->cancel(); break;
        case 2: _t->displayHeightField((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->addLinearHeightField(); break;
        case 4: _t->addGaussianHeightField(); break;
        case 5: _t->addPeriodicHeightField(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData WindowConfiguration_CombinedHeightField::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject WindowConfiguration_CombinedHeightField::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_WindowConfiguration_CombinedHeightField,
      qt_meta_data_WindowConfiguration_CombinedHeightField, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &WindowConfiguration_CombinedHeightField::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *WindowConfiguration_CombinedHeightField::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *WindowConfiguration_CombinedHeightField::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_WindowConfiguration_CombinedHeightField))
        return static_cast<void*>(const_cast< WindowConfiguration_CombinedHeightField*>(this));
    return QWidget::qt_metacast(_clname);
}

int WindowConfiguration_CombinedHeightField::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 6)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 6;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
