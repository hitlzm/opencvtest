// framesource.h — 暴露 source 属性给 QML 绑定，替代 Timer 刷新 hack
#ifndef FRAMESOURCE_H
#define FRAMESOURCE_H

#include <QObject>
#include <QString>

class FrameSource : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString source READ source NOTIFY sourceChanged)
public:
    explicit FrameSource(QObject *parent = nullptr)
        : QObject(parent)
        , m_source(QStringLiteral("image://imageProvider/frame?0"))
    {}

    QString source() const { return m_source; }

    /// 调用此方法强制 QML 重新请求 ImageProvider
    void refresh() {
        m_source = QString("image://imageProvider/frame?%1").arg(++m_counter);
        emit sourceChanged();
    }

signals:
    void sourceChanged();

private:
    QString m_source;
    int m_counter = 0;
};

#endif // FRAMESOURCE_H
