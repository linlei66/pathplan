
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace std;
using namespace cv;

// ==================== 地图参数 ====================

const int ROWS = 20;
const int COLS = 30;
const int CELL_SIZE = 30;

const int WINDOW_WIDTH = COLS * CELL_SIZE;
const int WINDOW_HEIGHT = ROWS * CELL_SIZE;

const int STRAIGHT_COST = 10;
const int DIAGONAL_COST = 14;

const int INF = numeric_limits<int>::max();

// 八方向移动
const Point directions[8] = {
    Point(0, -1),   // 上
    Point(0, 1),    // 下
    Point(-1, 0),   // 左
    Point(1, 0),    // 右
    Point(-1, -1),  // 左上
    Point(1, -1),   // 右上
    Point(-1, 1),   // 左下
    Point(1, 1)     // 右下
};

// ==================== 节点结构 ====================

struct AStarNode {

    Point pos;

    int g;
    int h;
    int f;

    AStarNode(Point p, int gCost, int hCost)
        : pos(p), g(gCost), h(hCost), f(gCost + hCost) {
    }
};

// 优先队列：f 值越小，优先级越高
struct CompareNode {

    bool operator()(const AStarNode& a,
                    const AStarNode& b) const {

        return a.f > b.f;
    }
};

// ==================== 地图数据 ====================

struct MapData {

    vector<vector<int>> grid;

    Point start;
    Point goal;

    bool hasStart = false;
    bool hasGoal = false;

    vector<Point> path;
    vector<Point> smoothPathResult;
    vector<Point> explored;

    MapData() {
        grid.assign(ROWS, vector<int>(COLS, 0));
    }
};

// ==================== 启发函数 ====================

// 八方向移动使用 Octile Distance
int heuristic(Point a, Point b) {

    int dx = abs(a.x - b.x);
    int dy = abs(a.y - b.y);

    int mn = min(dx, dy);
    int mx = max(dx, dy);

    return DIAGONAL_COST * mn
         + STRAIGHT_COST * (mx - mn);
}

// ==================== 地图边界检测 ====================

bool isValid(int x, int y) {

    return x >= 0 && x < COLS
        && y >= 0 && y < ROWS;
}



bool isObstacle(const MapData& mapData, int x, int y) {

    if (!isValid(x, y)) {
        return true;
    }

    return mapData.grid[y][x] == 1;
}


bool lineOfSight(
    const MapData& mapData,
    Point a,
    Point b
) {

    int x0 = a.x;
    int y0 = a.y;

    int x1 = b.x;
    int y1 = b.y;

    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);

    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;

    int err = dx - dy;

    while (true) {

        // 当前栅格不能是障碍物
        if (isObstacle(mapData, x0, y0)) {
            return false;
        }

        if (x0 == x1 && y0 == y1) {
            break;
        }

        int e2 = 2 * err;

        int oldX = x0;
        int oldY = y0;

        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }

        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }

        // 对角线穿越时，额外检查相邻侧向栅格，
        // 防止从两个相邻障碍物的角落穿过去。
        if (x0 != oldX && y0 != oldY) {

            if (isObstacle(mapData, x0, oldY) ||
                isObstacle(mapData, oldX, y0)) {

                return false;
            }
        }
    }

    return true;
}

void AStar(MapData& mapData) {

    mapData.path.clear();
    mapData.explored.clear();
    mapData.smoothPathResult.clear();

    if (!mapData.hasStart || !mapData.hasGoal) {
        cout << "请先设置起点和终点！" << endl;
        return;
    }

    if (isObstacle(mapData, mapData.start.x, mapData.start.y) ||
        isObstacle(mapData, mapData.goal.x, mapData.goal.y)) {

        cout << "起点或终点不能位于障碍物上！" << endl;
        return;
    }

    vector<vector<int>> gScore(
        ROWS, vector<int>(COLS, INF)
    );

    vector<vector<Point>> parent(
        ROWS, vector<Point>(COLS, Point(-1, -1))
    );

    vector<vector<bool>> closed(
        ROWS, vector<bool>(COLS, false)
    );

    priority_queue<
        AStarNode,
        vector<AStarNode>,
        CompareNode
    > openList;

    Point start = mapData.start;
    Point goal = mapData.goal;

    gScore[start.y][start.x] = 0;

    openList.push(
        AStarNode(
            start,
            0,
            heuristic(start, goal)
        )
    );

    bool found = false;

    // ==================== 搜索过程 ====================

    while (!openList.empty()) {

        AStarNode current = openList.top();
        openList.pop();

        Point p = current.pos;

        int x = p.x;
        int y = p.y;

        // 跳过已经处理过的旧节点
        if (closed[y][x]) {
            continue;
        }

        closed[y][x] = true;

        mapData.explored.push_back(p);

        // 到达终点
        if (p == goal) {
            found = true;
            break;
        }

        // 遍历八个方向
        for (int i = 0; i < 8; i++) {

            int nx = x + directions[i].x;
            int ny = y + directions[i].y;

            if (!isValid(nx, ny)) {
                continue;
            }

            if (isObstacle(mapData, nx, ny)) {
                continue;
            }

            if (closed[ny][nx]) {
                continue;
            }

            // 判断是否为对角线移动
            bool diagonal =
                directions[i].x != 0 &&
                directions[i].y != 0;

            // 防止斜着穿过障碍物角落
            if (diagonal) {

                if (isObstacle(mapData, nx, y) ||
                    isObstacle(mapData, x, ny)) {

                    continue;
                }
            }

            int moveCost = diagonal
                         ? DIAGONAL_COST
                         : STRAIGHT_COST;

            int tentativeG = gScore[y][x] + moveCost;

            if (tentativeG < gScore[ny][nx]) {

                gScore[ny][nx] = tentativeG;

                parent[ny][nx] = p;

                int h = heuristic(
                    Point(nx, ny),
                    goal
                );

                openList.push(
                    AStarNode(
                        Point(nx, ny),
                        tentativeG,
                        h
                    )
                );
            }
        }
    }

    // ==================== 路径回溯 ====================

    if (!found) {

        cout << "没有找到可行路径！" << endl;
        return;
    }

    Point current = goal;

    while (current != start) {

        mapData.path.push_back(current);

        current = parent[current.y][current.x];

        if (current.x == -1 || current.y == -1) {

            mapData.path.clear();

            cout << "路径回溯失败！" << endl;
            return;
        }
    }

    mapData.path.push_back(start);

    reverse(
        mapData.path.begin(),
        mapData.path.end()
    );

    cout << "A* 搜索成功！" << endl;
    cout << "原始路径节点数："
         << mapData.path.size() << endl;
}

// ==================== 路径平滑 ====================

vector<Point> smoothPath(
    const MapData& mapData,
    const vector<Point>& path
) {

    vector<Point> result;

    if (path.empty()) {
        return result;
    }

    if (path.size() <= 2) {
        return path;
    }

    // 从起点开始
    int currentIndex = 0;

    result.push_back(path[currentIndex]);

    while (currentIndex < static_cast<int>(path.size()) - 1) {

        // 从路径终点开始向前寻找
        int farthestIndex =
            static_cast<int>(path.size()) - 1;

        // 找到当前点能够直线到达的最远节点
        while (farthestIndex > currentIndex + 1) {

            if (lineOfSight(
                    mapData,
                    path[currentIndex],
                    path[farthestIndex]
                )) {

                break;
            }

            farthestIndex--;
        }

        // 添加最远可见节点
        result.push_back(path[farthestIndex]);

        currentIndex = farthestIndex;
    }

    return result;
}

// ==================== OpenCV 绘图 ====================

void drawMap(MapData& mapData) {

    Mat image(
        WINDOW_HEIGHT,
        WINDOW_WIDTH,
        CV_8UC3,
        Scalar(255, 255, 255)
    );

    // 绘制障碍物
    for (int y = 0; y < ROWS; y++) {

        for (int x = 0; x < COLS; x++) {

            if (mapData.grid[y][x] == 1) {

                rectangle(
                    image,
                    Point(
                        x * CELL_SIZE,
                        y * CELL_SIZE
                    ),
                    Point(
                        (x + 1) * CELL_SIZE,
                        (y + 1) * CELL_SIZE
                    ),
                    Scalar(50, 50, 50),
                    FILLED
                );
            }
        }
    }

    // 绘制搜索过的节点
    for (const Point& p : mapData.explored) {

        rectangle(
            image,
            Point(
                p.x * CELL_SIZE,
                p.y * CELL_SIZE
            ),
            Point(
                (p.x + 1) * CELL_SIZE,
                (p.y + 1) * CELL_SIZE
            ),
            Scalar(230, 240, 220),
            FILLED
        );
    }

    // 绘制网格
    for (int x = 0; x <= COLS; x++) {

        line(
            image,
            Point(x * CELL_SIZE, 0),
            Point(x * CELL_SIZE, WINDOW_HEIGHT),
            Scalar(210, 210, 210),
            1
        );
    }

    for (int y = 0; y <= ROWS; y++) {

        line(
            image,
            Point(0, y * CELL_SIZE),
            Point(WINDOW_WIDTH, y * CELL_SIZE),
            Scalar(210, 210, 210),
            1
        );
    }

    // 绘制原始 A* 路径
    for (size_t i = 1; i < mapData.path.size(); i++) {

        Point p1(
            mapData.path[i - 1].x * CELL_SIZE + CELL_SIZE / 2,
            mapData.path[i - 1].y * CELL_SIZE + CELL_SIZE / 2
        );

        Point p2(
            mapData.path[i].x * CELL_SIZE + CELL_SIZE / 2,
            mapData.path[i].y * CELL_SIZE + CELL_SIZE / 2
        );

        line(
            image,
            p1,
            p2,
            Scalar(0, 180, 255),
            2
        );
    }

    // 绘制平滑后的路径
    for (size_t i = 1;
         i < mapData.smoothPathResult.size();
         i++) {

        Point p1(
            mapData.smoothPathResult[i - 1].x * CELL_SIZE
                + CELL_SIZE / 2,
            mapData.smoothPathResult[i - 1].y * CELL_SIZE
                + CELL_SIZE / 2
        );

        Point p2(
            mapData.smoothPathResult[i].x * CELL_SIZE
                + CELL_SIZE / 2,
            mapData.smoothPathResult[i].y * CELL_SIZE
                + CELL_SIZE / 2
        );

        line(
            image,
            p1,
            p2,
            Scalar(0, 0, 255),
            3
        );
    }

    // 绘制起点
    if (mapData.hasStart) {

        circle(
            image,
            Point(
                mapData.start.x * CELL_SIZE + CELL_SIZE / 2,
                mapData.start.y * CELL_SIZE + CELL_SIZE / 2
            ),
            8,
            Scalar(0, 255, 0),
            FILLED
        );
    }

    // 绘制终点
    if (mapData.hasGoal) {

        circle(
            image,
            Point(
                mapData.goal.x * CELL_SIZE + CELL_SIZE / 2,
                mapData.goal.y * CELL_SIZE + CELL_SIZE / 2
            ),
            8,
            Scalar(255, 0, 0),
            FILLED
        );
    }

    imshow("A* Path Smoothing", image);
}

// ==================== 鼠标事件 ====================

void mouseCallback(
    int event,
    int x,
    int y,
    int flags,
    void* userdata
) {

    MapData* mapData =
        static_cast<MapData*>(userdata);

    if (event != EVENT_LBUTTONDOWN) {
        return;
    }

    int gridX = x / CELL_SIZE;
    int gridY = y / CELL_SIZE;

    if (!isValid(gridX, gridY)) {
        return;
    }

    // 按顺序设置起点、终点
    if (!mapData->hasStart) {

        if (isObstacle(*mapData, gridX, gridY)) {
            return;
        }

        mapData->start = Point(gridX, gridY);
        mapData->hasStart = true;

        cout << "起点：("
             << gridX << ", " << gridY << ")" << endl;
    }
    else if (!mapData->hasGoal) {

        if (isObstacle(*mapData, gridX, gridY)) {
            return;
        }

        mapData->goal = Point(gridX, gridY);
        mapData->hasGoal = true;

        cout << "终点：("
             << gridX << ", " << gridY << ")" << endl;
    }
    else {

        // 设置障碍物
        if (Point(gridX, gridY) == mapData->start ||
            Point(gridX, gridY) == mapData->goal) {
            return;
        }

        mapData->grid[gridY][gridX] =
            1 - mapData->grid[gridY][gridX];
    }

    mapData->path.clear();
    mapData->smoothPathResult.clear();
    mapData->explored.clear();

    drawMap(*mapData);
}

// ==================== 主函数 ====================

int main() {

    MapData mapData;

    namedWindow(
        "A* Path Smoothing",
        WINDOW_AUTOSIZE
    );

    setMouseCallback(
        "A* Path Smoothing",
        mouseCallback,
        &mapData
    );

    drawMap(mapData);

    cout << "========== A* 路径平滑 ==========" << endl;
    cout << "鼠标左键第一次点击：设置起点" << endl;
    cout << "鼠标左键第二次点击：设置终点" << endl;
    cout << "后续点击：添加或删除障碍物" << endl;
    cout << "按 S：开始 A* 搜索和平滑" << endl;
    cout << "按 R：重置地图" << endl;
    cout << "按 ESC：退出程序" << endl;

    while (true) {

        int key = waitKey(30);

        // 开始搜索和平滑
        if (key == 's' || key == 'S') {

            AStar(mapData);

            if (!mapData.path.empty()) {

                mapData.smoothPathResult =
                    smoothPath(
                        mapData,
                        mapData.path
                    );

                cout << "平滑后路径节点数："
                     << mapData.smoothPathResult.size()
                     << endl;
            }

            drawMap(mapData);
        }

        // 重置地图
        if (key == 'r' || key == 'R') {

            mapData = MapData();

            drawMap(mapData);

            cout << "地图已重置！" << endl;
        }

        // 退出
        if (key == 27) {
            break;
        }
    }

    destroyAllWindows();

    return 0;
}