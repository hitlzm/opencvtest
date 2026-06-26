#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QThread>
#include <iostream>
#include "opencv/videoprocessor.h"
#include "opencvimage/imageprovider.h"
#include "opencvimage/framesource.h"

int main(int argc, char *argv[])
{
    std::cout << "[main] Step 1: Starting..." << std::endl;

    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QGuiApplication app(argc, argv);

    // ── 1. 图像提供者（供 QML 显示） ──
    ImageProvider *provider = new ImageProvider();

    // ── 2. FrameSource（QML 属性绑定，替代 Timer 刷新） ──
    FrameSource *frameSource = new FrameSource();
    

    // ── 3. 视频处理器（工作线程） ──
    QThread *videoThread = new QThread;
    VideoProcessor *processor = new VideoProcessor;

    if (!processor->init("E:\\tank.mp4", "E:\\tank.png")) {
        std::cerr << "[main] VideoProcessor init FAILED" << std::endl;
        delete processor;
        delete videoThread;
        return -1;
    }

    processor->moveToThread(videoThread);

    // 工作线程每帧 → 主线程: ① 更新 ImageProvider  ② 触发 QML 刷新
    QObject::connect(processor, &VideoProcessor::frameReady, qApp,
                     [provider, frameSource](const QImage &img) {
        provider->updateImage(img);
        frameSource->refresh();         // QML 属性绑定自动触发 requestImage
    });

    QObject::connect(processor, &VideoProcessor::error,
                     [](const QString &msg) {
        std::cerr << "[VideoProcessor] " << msg.toStdString() << std::endl;
    });

    QObject::connect(videoThread, &QThread::started,
                     processor, &VideoProcessor::start);

    QObject::connect(processor, &VideoProcessor::finished,
                     videoThread, &QThread::quit);

    QObject::connect(videoThread, &QThread::finished,
                     processor, &QObject::deleteLater);
    QObject::connect(videoThread, &QThread::finished,
                     videoThread, &QObject::deleteLater);

    videoThread->start();
    std::cout << "[main] Step 2: Video thread started" << std::endl;

    // ── 4. QML 界面 ──
    QQmlApplicationEngine engine;

    // 注册 ImageProvider（供 image:// 协议）
    engine.addImageProvider(QLatin1String("imageProvider"), provider);

    // 将 FrameSource 暴露给 QML 根上下文（属性绑定）
    engine.rootContext()->setContextProperty(
        QStringLiteral("frameSource"), frameSource);

    const QUrl url(QStringLiteral("qrc:/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl) {
            std::cerr << "[main] QML object creation FAILED!" << std::endl;
            QCoreApplication::exit(-1);
        }
    }, Qt::QueuedConnection);

    engine.load(url);
    std::cout << "[main] Step 3: QML loaded" << std::endl;

    // 优雅关闭
    QObject::connect(&app, &QCoreApplication::aboutToQuit, [processor, videoThread]() {
        processor->stop();
        videoThread->quit();
    });

    std::cout << "[main] Step 4: Entering event loop" << std::endl;
    return app.exec();
}
