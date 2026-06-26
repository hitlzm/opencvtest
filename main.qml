import QtQuick 2.12
import QtQuick.Window 2.12

Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Video Object Detection")

    Image {
        id: videoDisplay
        anchors.fill: parent
        fillMode: Image.PreserveAspectFit
        // 直接绑定 FrameSource 的 source 属性
        // 每次 refresh() 改变 counter → NOTIFY 信号触发 → QML 自动重新请求
        source: frameSource.source
    }
}
