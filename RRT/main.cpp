
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>

using namespace std;
using namespace cv;

// ======================================================
// 1. 地图参数和 RRT 参数
// ======================================================

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
//每次最多移动20个像素
const float STEP_SIZE = 20.0f;
//终点连接半径
const float GOAL_RADIUS = 30.0f;
//最大迭代次数
const int MAX_ITERATIONS = 10000;
//目标偏置概率，10% 的概率直接采样终点
const float GOAL_BIAS = 0.10f;
//浮点数误差容忍值
const float EPS = 1e-6f;

// ======================================================
// 2. 障碍物数据结构
// ======================================================

// 矩形障碍物
struct RectangleObstacle {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};

// 圆形障碍物
struct CircleObstacle {
    Point2f center;
    float radius;
};

// 三角形障碍物
struct TriangleObstacle {
    Point2f p1;
    Point2f p2;
    Point2f p3;
};

// ======================================================
// 3. RRT 节点
// ======================================================

struct RRTNode {
    Point2f pos;
    //当前节点的父节点在 RRT 搜索树中的索引
    int parent;

    RRTNode(Point2f p, int par) {
        pos = p;
        parent = par;
    }
};

// ======================================================
// 4. 绘制模式
// ======================================================

enum class DrawMode {
    SetStart,
    SetGoal,
    Rectangle,
    Circle,
    Triangle
};

// ======================================================
// 5. 地图数据
// ======================================================

struct MapData {

    // 起点和终点
    Point2f start;
    Point2f goal;

    bool hasStart = false;
    bool hasGoal = false;

    // 障碍物
    vector<RectangleObstacle> rectangles;
    vector<CircleObstacle> circles;
    vector<TriangleObstacle> triangles;

    // RRT 搜索树和最终路径
    vector<RRTNode> tree;
    vector<Point2f> path;

    // 当前绘制模式
    DrawMode mode = DrawMode::SetStart;

    // 鼠标拖动状态
    bool isDragging = false;
    //鼠标开始拖动时的位置
    Point2f dragStart;
    //鼠标当前所在的位置
    Point2f cursor;

    // 三角形已经点击的顶点
    vector<Point2f> pendingTriangle;
};

// ======================================================
// 6. 基础几何计算
// ======================================================

// 计算两点之间的距离
float distance(Point2f a, Point2f b) {

    float dx = a.x - b.x;
    float dy = a.y - b.y;

    return sqrt(dx * dx + dy * dy);
}

// 计算向量叉积
float cross(
    Point2f a,
    Point2f b,
    Point2f c
) {
    return (b.x - a.x) * (c.y - a.y)
         - (b.y - a.y) * (c.x - a.x);
}

// 判断点是否在线段上
bool pointOnSegment(
    Point2f p,
    Point2f a,
    Point2f b
) {

    if (abs(cross(a, b, p)) > EPS) {
        return false;
    }

    return p.x >= min(a.x, b.x) - EPS &&
           p.x <= max(a.x, b.x) + EPS &&
           p.y >= min(a.y, b.y) - EPS &&
           p.y <= max(a.y, b.y) + EPS;
}

// ======================================================
// 7. 线段与线段相交检测
// ======================================================

bool segmentsIntersect(
    Point2f a,
    Point2f b,
    Point2f c,
    Point2f d
) {

    float c1 = cross(a, b, c);
    float c2 = cross(a, b, d);
    float c3 = cross(c, d, a);
    float c4 = cross(c, d, b);

    // 普通相交
    if (((c1 > EPS && c2 < -EPS) ||
         (c1 < -EPS && c2 > EPS)) &&
        ((c3 > EPS && c4 < -EPS) ||
         (c3 < -EPS && c4 > EPS))) {

        return true;
    }

    // 处理端点接触和共线情况
    if (abs(c1) <= EPS && pointOnSegment(c, a, b))
        return true;

    if (abs(c2) <= EPS && pointOnSegment(d, a, b))
        return true;

    if (abs(c3) <= EPS && pointOnSegment(a, c, d))
        return true;

    if (abs(c4) <= EPS && pointOnSegment(b, c, d))
        return true;

    return false;
}

// ======================================================
// 8. 点与矩形碰撞检测
// ======================================================

bool pointInRectangle(
    Point2f p,
    const RectangleObstacle& rect
) {

    return p.x >= rect.xmin - EPS &&
           p.x <= rect.xmax + EPS &&
           p.y >= rect.ymin - EPS &&
           p.y <= rect.ymax + EPS;
}

// ======================================================
// 9. 线段与矩形碰撞检测
// ======================================================

// Liang-Barsky 线段裁剪算法
// 碰到矩形边界也视为碰撞

bool segmentIntersectsRectangle(
    Point2f a,
    Point2f b,
    const RectangleObstacle& rect
) {

    float dx = b.x - a.x;
    float dy = b.y - a.y;

    float tMin = 0.0f;
    float tMax = 1.0f;

    auto clip = [&](float p, float q) {

        if (abs(p) < EPS) {
            return q >= -EPS;
        }

        float r = q / p;

        if (p < 0) {
            tMin = max(tMin, r);
        }
        else {
            tMax = min(tMax, r);
        }

        return tMin <= tMax + EPS;
    };

    if (!clip(-dx, a.x - rect.xmin))
        return false;

    if (!clip(dx, rect.xmax - a.x))
        return false;

    if (!clip(-dy, a.y - rect.ymin))
        return false;

    if (!clip(dy, rect.ymax - a.y))
        return false;

    return true;
}

// ======================================================
// 10. 点到线段的最短距离
// ======================================================

float pointToSegmentDistance(
    Point2f p,
    Point2f a,
    Point2f b
) {

    float dx = b.x - a.x;
    float dy = b.y - a.y;

    float lengthSquared = dx * dx + dy * dy;

    if (lengthSquared < EPS) {
        return distance(p, a);
    }

    float t =
        ((p.x - a.x) * dx +
         (p.y - a.y) * dy) / lengthSquared;

    t = max(0.0f, min(1.0f, t));

    Point2f nearest(
        a.x + t * dx,
        a.y + t * dy
    );

    return distance(p, nearest);
}

// ======================================================
// 11. 点与圆形碰撞检测
// ======================================================

bool pointInCircle(
    Point2f p,
    const CircleObstacle& circle
) {

    return distance(p, circle.center)
           <= circle.radius + EPS;
}

// ======================================================
// 12. 线段与圆形碰撞检测
// ======================================================

bool segmentIntersectsCircle(
    Point2f a,
    Point2f b,
    const CircleObstacle& circle
) {

    float d = pointToSegmentDistance(
        circle.center,
        a,
        b
    );

    return d <= circle.radius + EPS;
}

// ======================================================
// 13. 点与三角形碰撞检测
// ======================================================

// 使用叉积判断点是否在三角形内部或边界上

bool pointInTriangle(
    Point2f p,
    const TriangleObstacle& tri
) {

    float c1 = cross(tri.p1, tri.p2, p);
    float c2 = cross(tri.p2, tri.p3, p);
    float c3 = cross(tri.p3, tri.p1, p);

    bool hasNegative =
        c1 < -EPS || c2 < -EPS || c3 < -EPS;

    bool hasPositive =
        c1 > EPS || c2 > EPS || c3 > EPS;

    return !(hasNegative && hasPositive);
}

// ======================================================
// 14. 线段与三角形碰撞检测
// ======================================================

bool segmentIntersectsTriangle(
    Point2f a,
    Point2f b,
    const TriangleObstacle& tri
) {

    // 端点在三角形内部或边界上
    if (pointInTriangle(a, tri) ||
        pointInTriangle(b, tri)) {

        return true;
    }

    // 检查线段与三条边是否相交
    if (segmentsIntersect(
        a, b, tri.p1, tri.p2
    )) {
        return true;
    }

    if (segmentsIntersect(
        a, b, tri.p2, tri.p3
    )) {
        return true;
    }

    if (segmentsIntersect(
        a, b, tri.p3, tri.p1
    )) {
        return true;
    }

    return false;
}

// ======================================================
// 15. 检查一个点是否在地图范围内且没有碰撞
// ======================================================

bool isFreePoint(
    const MapData& mapData,
    Point2f p
) {

    // 地图边界
    if (p.x < 0 || p.y < 0 ||
        p.x >= WINDOW_WIDTH ||
        p.y >= WINDOW_HEIGHT) {

        return false;
    }

    // 检查矩形
    for (const auto& rect : mapData.rectangles) {

        if (pointInRectangle(p, rect)) {
            return false;
        }
    }

    // 检查圆形
    for (const auto& circle : mapData.circles) {

        if (pointInCircle(p, circle)) {
            return false;
        }
    }

    // 检查三角形
    for (const auto& tri : mapData.triangles) {

        if (pointInTriangle(p, tri)) {
            return false;
        }
    }

    return true;
}

// ======================================================
// 16. 检查一条线段是否与障碍物碰撞
// ======================================================

bool collisionFree(
    const MapData& mapData,
    //nearest,randomPoint
    Point2f a,
    Point2f b
) {

    // 检查线段两个端点
    if (!isFreePoint(mapData, a) ||
        !isFreePoint(mapData, b)) {

        return false;
    }

    // 检查矩形
    for (const auto& rect : mapData.rectangles) {

        if (segmentIntersectsRectangle(a, b, rect)) {
            return false;
        }
    }

    // 检查圆形
    for (const auto& circle : mapData.circles) {

        if (segmentIntersectsCircle(a, b, circle)) {
            return false;
        }
    }

    // 检查三角形
    for (const auto& tri : mapData.triangles) {

        if (segmentIntersectsTriangle(a, b, tri)) {
            return false;
        }
    }

    return true;
}

// ======================================================
// 17. 寻找距离随机点最近的树节点
// ======================================================

int findNearestNode(
    const vector<RRTNode>& tree,
    Point2f randomPoint
) {

    int nearestIndex = 0;

    float minDist = distance(
        tree[0].pos,
        randomPoint
    );

    for (int i = 1;
         i < static_cast<int>(tree.size());
         i++) {

        float d = distance(
            tree[i].pos,
            randomPoint
        );

        if (d < minDist) {

            minDist = d;

            nearestIndex = i;
        }
    }

    return nearestIndex;
}

// ======================================================
// 18. 从最近节点向随机点扩展
// ======================================================

Point2f steer(
    Point2f nearest,
    Point2f randomPoint
) {

    float dist = distance(
        nearest,
        randomPoint
    );

    if (dist <= STEP_SIZE) {
        return randomPoint;
    }

    float dx = randomPoint.x - nearest.x;
    float dy = randomPoint.y - nearest.y;

    float scale = STEP_SIZE / dist;

    return Point2f(
        nearest.x + dx * scale,
        nearest.y + dy * scale
    );
}

// ======================================================
// 19. 路径恢复
// ======================================================

void reconstructPath(
    MapData& mapData,
    int goalIndex
) {

    mapData.path.clear();

    int current = goalIndex;

    while (current != -1) {

        mapData.path.push_back(
            mapData.tree[current].pos
        );

        current = mapData.tree[current].parent;
    }

    reverse(
        mapData.path.begin(),
        mapData.path.end()
    );
}

// ======================================================
// 20. RRT 算法
// ======================================================

void RRT(MapData& mapData) {

    mapData.tree.clear();
    mapData.path.clear();

    if (!mapData.hasStart || !mapData.hasGoal) {

        cout << "请先设置起点和终点！" << endl;
        return;
    }

    Point2f start = mapData.start;
    Point2f goal = mapData.goal;

    if (!isFreePoint(mapData, start) ||
        !isFreePoint(mapData, goal)) {

        cout << "起点或终点发生碰撞！" << endl;
        return;
    }

    if (distance(start, goal) < EPS) {

        cout << "起点和终点不能重合！" << endl;
        return;
    }

    // 随机数生成器
    random_device rd;
    mt19937 gen(rd());
    //创建三个随机数分布器，分别用于生成随机 X 坐标、随机 Y 坐标，以及 0～1 之间的随机概率
    uniform_real_distribution<float> randX(
        0.0f,
        WINDOW_WIDTH - 0.001f
    );

    uniform_real_distribution<float> randY(
        0.0f,
        WINDOW_HEIGHT - 0.001f
    );

    uniform_real_distribution<float> probability(
        0.0f,
        1.0f
    );

    // 起点作为搜索树根节点
    mapData.tree.push_back(
        RRTNode(start, -1)
    );

    cout << "\n========== RRT 开始搜索 ==========" << endl;

    bool found = false;

    // 迭代搜索
    for (int iter = 0;
         iter < MAX_ITERATIONS;
         iter++) {

        Point2f randomPoint;

        // Goal Bias：一定概率直接采样终点
        if (probability(gen) < GOAL_BIAS) {

            randomPoint = goal;
        }
        else {

            randomPoint = Point2f(
                randX(gen),
                randY(gen)
            );
        }

        // 1. 寻找最近节点
        int nearestIndex = findNearestNode(
            mapData.tree,
            randomPoint
        );

        Point2f nearest =
            mapData.tree[nearestIndex].pos;

        // 2. 向随机点扩展
        Point2f newPoint = steer(
            nearest,
            randomPoint
        );

        // 避免生成重复节点
        if (distance(nearest, newPoint) < EPS) {
            continue;
        }

        // 3. 检查新节点和新边
        if (!collisionFree(
            mapData,
            nearest,
            newPoint
        )) {
            continue;
        }

        // 4. 将新节点加入搜索树
        mapData.tree.push_back(
            RRTNode(newPoint, nearestIndex)
        );

        int newIndex =
            static_cast<int>(mapData.tree.size()) - 1;

        // 5. 检查是否能够连接终点
        float goalDist = distance(
            newPoint,
            goal
        );

        if (goalDist <= GOAL_RADIUS &&
            collisionFree(mapData, newPoint, goal)) {

            // 将终点作为新节点加入树
            mapData.tree.push_back(
                RRTNode(goal, newIndex)
            );

            int goalIndex =
                static_cast<int>(mapData.tree.size()) - 1;

            // 6. 恢复路径
            reconstructPath(
                mapData,
                goalIndex
            );

            found = true;

            cout << "找到可行路径！" << endl;
            cout << "迭代次数：" << iter + 1 << endl;
            cout << "搜索树节点数："
                 << mapData.tree.size() << endl;

            break;
        }
    }

    if (!found) {

        cout << "未找到路径！" << endl;
        cout << "达到最大迭代次数："
             << MAX_ITERATIONS << endl;
    }

    cout << "==================================" << endl;
}

// ======================================================
// 21. 判断鼠标位置是否位于障碍物中
// 用于右键删除障碍物
// ======================================================

bool pointInObstacle(
    const MapData& mapData,
    Point2f p
) {

    for (const auto& rect : mapData.rectangles) {

        if (pointInRectangle(p, rect)) {
            return true;
        }
    }

    for (const auto& circle : mapData.circles) {

        if (pointInCircle(p, circle)) {
            return true;
        }
    }

    for (const auto& tri : mapData.triangles) {

        if (pointInTriangle(p, tri)) {
            return true;
        }
    }

    return false;
}

// ======================================================
// 22. 清除搜索结果
// ======================================================

void clearSearch(MapData& mapData) {

    mapData.tree.clear();
    mapData.path.clear();
}

// ======================================================
// 23. 获取当前模式名称
// ======================================================

string getModeName(DrawMode mode) {

    switch (mode) {

    case DrawMode::SetStart:
        return "Set Start";

    case DrawMode::SetGoal:
        return "Set Goal";

    case DrawMode::Rectangle:
        return "Rectangle";

    case DrawMode::Circle:
        return "Circle";

    case DrawMode::Triangle:
        return "Triangle";
    }

    return "";
}

// ======================================================
// 24. 地图绘制函数
// ======================================================

void drawMap(const MapData& mapData) {

    Mat image(
        WINDOW_HEIGHT,
        WINDOW_WIDTH,
        CV_8UC3,
        Scalar(255, 255, 255)
    );

    // ---------------- 绘制 RRT 搜索树 ----------------

    for (int i = 1;
         i < static_cast<int>(mapData.tree.size());
         i++) {

        int parent = mapData.tree[i].parent;

        Point p1(
            cvRound(mapData.tree[parent].pos.x),
            cvRound(mapData.tree[parent].pos.y)
        );

        Point p2(
            cvRound(mapData.tree[i].pos.x),
            cvRound(mapData.tree[i].pos.y)
        );

        line(
            image,
            p1,
            p2,
            Scalar(190, 190, 190),
            1,
            LINE_AA
        );
    }

    // ---------------- 绘制矩形障碍物 ----------------

    for (const auto& rect : mapData.rectangles) {

        rectangle(
            image,
            Point(
                cvFloor(rect.xmin),
                cvFloor(rect.ymin)
            ),
            Point(
                cvCeil(rect.xmax),
                cvCeil(rect.ymax)
            ),
            Scalar(30, 30, 30),
            FILLED
        );
    }

    // ---------------- 绘制圆形障碍物 ----------------

    for (const auto& circle : mapData.circles) {

        cv::circle(
            image,
            Point(
                cvRound(circle.center.x),
                cvRound(circle.center.y)
            ),
            cvRound(circle.radius),
            Scalar(30, 30, 30),
            FILLED,
            LINE_AA
        );
    }

    // ---------------- 绘制三角形障碍物 ----------------

    for (const auto& tri : mapData.triangles) {

        vector<Point> points = {

            Point(
                cvRound(tri.p1.x),
                cvRound(tri.p1.y)
            ),

            Point(
                cvRound(tri.p2.x),
                cvRound(tri.p2.y)
            ),

            Point(
                cvRound(tri.p3.x),
                cvRound(tri.p3.y)
            )
        };

        fillConvexPoly(
            image,
            points,
            Scalar(30, 30, 30),
            LINE_AA
        );
    }

    // ---------------- 绘制最终路径 ----------------

    for (int i = 1;
         i < static_cast<int>(mapData.path.size());
         i++) {

        Point p1(
            cvRound(mapData.path[i - 1].x),
            cvRound(mapData.path[i - 1].y)
        );

        Point p2(
            cvRound(mapData.path[i].x),
            cvRound(mapData.path[i].y)
        );

        line(
            image,
            p1,
            p2,
            Scalar(0, 220, 255),
            4,
            LINE_AA
        );
    }

    // ---------------- 绘制起点 ----------------

    if (mapData.hasStart) {

        circle(
            image,
            Point(
                cvRound(mapData.start.x),
                cvRound(mapData.start.y)
            ),
            7,
            Scalar(0, 200, 0),
            FILLED,
            LINE_AA
        );
    }

    // ---------------- 绘制终点 ----------------

    if (mapData.hasGoal) {

        circle(
            image,
            Point(
                cvRound(mapData.goal.x),
                cvRound(mapData.goal.y)
            ),
            7,
            Scalar(0, 0, 255),
            FILLED,
            LINE_AA
        );
    }

    // ---------------- 绘制矩形预览 ----------------

    if (mapData.isDragging &&
        mapData.mode == DrawMode::Rectangle) {

        rectangle(
            image,
            Point(
                cvRound(min(mapData.dragStart.x, mapData.cursor.x)),
                cvRound(min(mapData.dragStart.y, mapData.cursor.y))
            ),
            Point(
                cvRound(max(mapData.dragStart.x, mapData.cursor.x)),
                cvRound(max(mapData.dragStart.y, mapData.cursor.y))
            ),
            Scalar(255, 100, 0),
            2
        );
    }

    // ---------------- 绘制圆形预览 ----------------

    if (mapData.isDragging &&
        mapData.mode == DrawMode::Circle) {

        int radius = cvRound(
            distance(mapData.dragStart, mapData.cursor)
        );

        circle(
            image,
            Point(
                cvRound(mapData.dragStart.x),
                cvRound(mapData.dragStart.y)
            ),
            radius,
            Scalar(255, 100, 0),
            2
        );
    }

    // ---------------- 绘制三角形预览 ----------------

    if (!mapData.pendingTriangle.empty()) {

        for (int i = 0;
             i < static_cast<int>(mapData.pendingTriangle.size());
             i++) {

            Point p(
                cvRound(mapData.pendingTriangle[i].x),
                cvRound(mapData.pendingTriangle[i].y)
            );

            circle(
                image,
                p,
                4,
                Scalar(255, 100, 0),
                FILLED
            );

            if (i > 0) {

                Point previous(
                    cvRound(mapData.pendingTriangle[i - 1].x),
                    cvRound(mapData.pendingTriangle[i - 1].y)
                );

                line(
                    image,
                    previous,
                    p,
                    Scalar(255, 100, 0),
                    2
                );
            }
        }

        if (mapData.pendingTriangle.size() == 2) {

            Point p1(
                cvRound(mapData.pendingTriangle[1].x),
                cvRound(mapData.pendingTriangle[1].y)
            );

            Point p2(
                cvRound(mapData.cursor.x),
                cvRound(mapData.cursor.y)
            );

            line(
                image,
                p1,
                p2,
                Scalar(255, 100, 0),
                2
            );
        }
    }

    // ---------------- 绘制模式提示 ----------------

    string modeText =
        "Mode: " + getModeName(mapData.mode);

    putText(
        image,
        modeText,
        Point(10, 25),
        FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar(80, 80, 80),
        1,
        LINE_AA
    );

    imshow("RRT Continuous Space", image);
}

// ======================================================
// 25. 鼠标回调函数
// ======================================================

void mouseCallback(
    int event,
    int x,
    int y,
    int flags,
    void* userdata
) {

    MapData* mapData =
        static_cast<MapData*>(userdata);

    if (x < 0 || y < 0 ||
        x >= WINDOW_WIDTH ||
        y >= WINDOW_HEIGHT) {

        return;
    }

    Point2f p(
        static_cast<float>(x),
        static_cast<float>(y)
    );

    mapData->cursor = p;

    // ---------------- 鼠标移动 ----------------

    if (event == EVENT_MOUSEMOVE) {

        if (mapData->isDragging ||
            !mapData->pendingTriangle.empty()) {

            drawMap(*mapData);
        }

        return;
    }

    // ---------------- 左键按下 ----------------

    if (event == EVENT_LBUTTONDOWN) {

        // 设置起点
        if (mapData->mode == DrawMode::SetStart) {

            if (!isFreePoint(*mapData, p)) {

                cout << "起点不能设置在障碍物上！" << endl;
                return;
            }

            mapData->start = p;
            mapData->hasStart = true;

            clearSearch(*mapData);

            cout << "起点：("
                 << x << ", " << y << ")" << endl;

            drawMap(*mapData);
            return;
        }

        // 设置终点
        if (mapData->mode == DrawMode::SetGoal) {

            if (!isFreePoint(*mapData, p)) {

                cout << "终点不能设置在障碍物上！" << endl;
                return;
            }

            if (mapData->hasStart &&
                distance(mapData->start, p) < EPS) {

                cout << "终点不能与起点重合！" << endl;
                return;
            }

            mapData->goal = p;
            mapData->hasGoal = true;

            clearSearch(*mapData);

            cout << "终点：("
                 << x << ", " << y << ")" << endl;

            drawMap(*mapData);
            return;
        }

        // 矩形或圆形：开始拖动
        if (mapData->mode == DrawMode::Rectangle ||
            mapData->mode == DrawMode::Circle) {

            mapData->dragStart = p;
            mapData->cursor = p;
            mapData->isDragging = true;

            drawMap(*mapData);
            return;
        }

        // 三角形：依次点击三个顶点
        if (mapData->mode == DrawMode::Triangle) {

            mapData->pendingTriangle.push_back(p);

            if (mapData->pendingTriangle.size() == 3) {

                TriangleObstacle tri;

                tri.p1 = mapData->pendingTriangle[0];
                tri.p2 = mapData->pendingTriangle[1];
                tri.p3 = mapData->pendingTriangle[2];

                // 检查三角形面积，避免三个点共线
                float area2 = abs(
                    cross(tri.p1, tri.p2, tri.p3)
                );

                if (area2 > EPS) {

                    mapData->triangles.push_back(tri);

                    clearSearch(*mapData);

                    cout << "已添加三角形障碍物！" << endl;
                }
                else {

                    cout << "三角形三个顶点不能共线！" << endl;
                }

                mapData->pendingTriangle.clear();
            }

            drawMap(*mapData);
            return;
        }
    }

    // ---------------- 左键松开 ----------------

    if (event == EVENT_LBUTTONUP &&
        mapData->isDragging) {

        mapData->cursor = p;

        if (mapData->mode == DrawMode::Rectangle) {

            RectangleObstacle rect;

            rect.xmin = min(
                mapData->dragStart.x,
                p.x
            );

            rect.ymin = min(
                mapData->dragStart.y,
                p.y
            );

            rect.xmax = max(
                mapData->dragStart.x,
                p.x
            );

            rect.ymax = max(
                mapData->dragStart.y,
                p.y
            );

            if (rect.xmax - rect.xmin >= 3 &&
                rect.ymax - rect.ymin >= 3) {

                mapData->rectangles.push_back(rect);

                clearSearch(*mapData);

                cout << "已添加矩形障碍物！" << endl;
            }
            else {

                cout << "矩形尺寸太小，未添加！" << endl;
            }
        }

        if (mapData->mode == DrawMode::Circle) {

            CircleObstacle circle;

            circle.center = mapData->dragStart;

            circle.radius = distance(
                mapData->dragStart,
                p
            );

            if (circle.radius >= 3) {

                mapData->circles.push_back(circle);

                clearSearch(*mapData);

                cout << "已添加圆形障碍物！" << endl;
            }
            else {

                cout << "圆形半径太小，未添加！" << endl;
            }
        }

        mapData->isDragging = false;

        drawMap(*mapData);
        return;
    }

    // ---------------- 右键删除障碍物 ----------------

    if (event == EVENT_RBUTTONDOWN) {

        bool deleted = false;

        // 从后往前检查，优先删除最上层障碍物

        for (int i = static_cast<int>(
                 mapData->triangles.size()
             ) - 1; i >= 0; i--) {

            if (pointInTriangle(
                p,
                mapData->triangles[i]
            )) {

                mapData->triangles.erase(
                    mapData->triangles.begin() + i
                );

                deleted = true;
                break;
            }
        }

        if (!deleted) {

            for (int i = static_cast<int>(
                     mapData->circles.size()
                 ) - 1; i >= 0; i--) {

                if (pointInCircle(
                    p,
                    mapData->circles[i]
                )) {

                    mapData->circles.erase(
                        mapData->circles.begin() + i
                    );

                    deleted = true;
                    break;
                }
            }
        }

        if (!deleted) {

            for (int i = static_cast<int>(
                     mapData->rectangles.size()
                 ) - 1; i >= 0; i--) {

                if (pointInRectangle(
                    p,
                    mapData->rectangles[i]
                )) {

                    mapData->rectangles.erase(
                        mapData->rectangles.begin() + i
                    );

                    deleted = true;
                    break;
                }
            }
        }

        if (deleted) {

            clearSearch(*mapData);

            cout << "已删除障碍物！" << endl;
        }
        else {

            cout << "当前位置没有障碍物！" << endl;
        }

        drawMap(*mapData);
    }
}

// ======================================================
// 26. 重置地图
// ======================================================

void resetMap(MapData& mapData) {

    mapData.hasStart = false;
    mapData.hasGoal = false;

    mapData.rectangles.clear();
    mapData.circles.clear();
    mapData.triangles.clear();
    mapData.tree.clear();
    mapData.path.clear();
    mapData.pendingTriangle.clear();
    mapData.isDragging = false;
    mapData.mode = DrawMode::SetStart;
    cout << "地图已重置！" << endl;
    drawMap(mapData);
}
// ======================================================
// 27. 主函数
// ======================================================

int main() {
    MapData mapData;
    namedWindow(
        "RRT Continuous Space",
        WINDOW_AUTOSIZE
    );
    setMouseCallback(
        "RRT Continuous Space",
        mouseCallback,
        &mapData
    );
    drawMap(mapData);
    cout << "\n========== RRT 连续空间路径规划 ==========" << endl;
    cout << "1：设置起点模式" << endl;
    cout << "2：设置终点模式" << endl;
    cout << "3：矩形障碍物模式" << endl;
    cout << "4：圆形障碍物模式" << endl;
    cout << "5：三角形障碍物模式" << endl;
    cout << "空格：开始 RRT 搜索" << endl;
    cout << "C：清除搜索树和路径" << endl;
    cout << "R：重置地图" << endl;
    cout << "右键：删除障碍物" << endl;
    cout << "ESC：退出程序" << endl;
    cout << "===========================================" << endl;
    while (true) {
        int key = waitKey(20);
        // ESC：退出
        if (key == 27) {
            break;
        }
        // 1：起点模式
        if (key == '1') {
            mapData.mode = DrawMode::SetStart;
            mapData.pendingTriangle.clear();
            mapData.isDragging = false;
            cout << "当前模式：设置起点" << endl;
            drawMap(mapData);
        }
        // 2：终点模式
        if (key == '2') {
            mapData.mode = DrawMode::SetGoal;
            mapData.pendingTriangle.clear();
            mapData.isDragging = false;
            cout << "当前模式：设置终点" << endl;
            drawMap(mapData);
        }
        // 3：矩形模式
        if (key == '3') {
            mapData.mode = DrawMode::Rectangle;
            mapData.pendingTriangle.clear();
            mapData.isDragging = false;
            cout << "当前模式：绘制矩形" << endl;
            drawMap(mapData);
        }
        // 4：圆形模式
        if (key == '4') {
            mapData.mode = DrawMode::Circle;
            mapData.pendingTriangle.clear();
            mapData.isDragging = false;
            cout << "当前模式：绘制圆形" << endl;
            drawMap(mapData);
        }
        // 5：三角形模式
        if (key == '5') {
            mapData.mode = DrawMode::Triangle;
            mapData.pendingTriangle.clear();
            mapData.isDragging = false;
            cout << "当前模式：绘制三角形，请点击三个顶点" << endl;
            drawMap(mapData);
        }
        // 空格：开始 RRT
        if (key == 32) {
            RRT(mapData);
            drawMap(mapData);
        }
        // C：清除搜索结果
        if (key == 'c' || key == 'C') {
            clearSearch(mapData);
            cout << "搜索树和路径已清除！" << endl;
            drawMap(mapData);
        }
        // R：重置地图
        if (key == 'r' || key == 'R') {
            resetMap(mapData);
        }
    }
    destroyAllWindows();
    return 0;
}