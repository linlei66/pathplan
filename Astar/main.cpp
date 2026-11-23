
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

// ==================== 地图参数 ====================

const int ROWS = 10;
const int COLS = 10;
const int CELL_SIZE = 20;

const int WINDOW_WIDTH = COLS * CELL_SIZE;
const int WINDOW_HEIGHT = ROWS * CELL_SIZE;

// 直线移动代价和斜向移动代价
const int STRAIGHT_COST = 10;
const int DIAGONAL_COST = 14;

// ==================== A* 节点 ====================

struct AStarNode {
    Point pos;
    int g;
    int h;
    int f;
    // 构造函数会根据当前点的 g 和 h 计算 f：
    // 初始化表达式	实际含义
    // pos(p)	       pos = p
    // g(gCost)	   g = gCost
    // h(hCost)	   h = hCost
    // f(gCost + hCost)	f = gCost + hCost
    AStarNode(Point p, int gCost, int hCost)
        : pos(p), g(gCost), h(hCost), f(gCost + hCost) {}
};

// 优先队列比较器
//比较器返回 true，表示第一个参数 a 的优先级低于第二个参数 b，所以 b 应该排在前面
struct CompareNode {
    bool operator()(const AStarNode& a,
                    const AStarNode& b) const {
        return a.f > b.f;
    }
};

// ==================== 地图数据 ====================

struct MapData {

    // 0：可通行
    // 1：障碍物
    vector<vector<int>> grid;

    Point start;
    Point goal;

    bool hasStart = false;
    bool hasGoal = false;
    // 保存最终路径
    vector<Point> path;
    //保存搜索过程中访问过的节点
    vector<Point> explored;
    //创建 MapData 对象时，自动初始化地图网格
    MapData() {
        grid.assign(ROWS, vector<int>(COLS, 0));
    }
};

// ==================== 启发函数 ====================

// 八方向搜索使用 Octile Distance
int heuristic(Point a, Point b) {
    //abs绝对值
    int dx = abs(a.x - b.x);
    int dy = abs(a.y - b.y);

    int mn = min(dx, dy);
    int mx = max(dx, dy);
//八方向移动时，可以先沿对角线走较短方向的全部距离，再沿剩余方向直线移动
    return DIAGONAL_COST * mn +
           STRAIGHT_COST * (mx - mn);
}

// ==================== 判断坐标是否合法 ====================

bool isValid(int x, int y) {

    return x >= 0 && x < COLS &&
           y >= 0 && y < ROWS;
}

// ==================== A* 搜索算法 ====================

void AStar(MapData& mapData) {

    mapData.path.clear();
    mapData.explored.clear();

    if (!mapData.hasStart || !mapData.hasGoal) {

        cout << "请先设置起点和终点！" << endl;
        return;
    }
    //C++ 的二维数组采用的是 先行后列（row, column） 的访问方式
    if (mapData.grid[mapData.start.y][mapData.start.x] == 1 ||
        mapData.grid[mapData.goal.y][mapData.goal.x] == 1) {

        cout << "起点或终点不能是障碍物！" << endl;
        return;
    }
    //一开始我们不知道从起点到每个节点的实际代价是多少，所以先把所有节点的代价设置为一个很大的数
    const int INF = 1e9;
    // 每个节点的实际代价
    //vector<int>(COLS, INF)意味着每一行都是一个长度为 COLS 每个元素为INF的向量
    vector<vector<int>> gScore(
        ROWS, vector<int>(COLS, INF));
    // 每个节点的父节点
    //这个二维数组保存每个节点是从哪个节点走过来的
    //parent[3][4] = Point(3, 3);表示到达 (4,3) 的上一节点是 (3,3)
    vector<vector<Point>> parent(
        ROWS,
        vector<Point>(COLS, Point(-1, -1)));
    // 是否已经扩展 可以防止算法反复处理已经完成搜索的节点
    vector<vector<bool>> closed(
        ROWS, vector<bool>(COLS, false));
    // OpenList
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

    // 八个移动方向
    const int dx[8] = {
        0, 0, -1, 1,
        -1, 1, -1, 1
    };

    const int dy[8] = {
        -1, 1, 0, 0,
        -1, -1, 1, 1
    };

    bool found = false;

    while (!openList.empty()) {

        AStarNode current = openList.top();
        openList.pop();

        Point p = current.pos;

        // 跳过已经处理的节点
        if (closed[p.y][p.x])
            continue;

        closed[p.y][p.x] = true;

        mapData.explored.push_back(p);

        // 到达终点
        if (p == goal) {

            found = true;
            break;
        }

        // 遍历八个方向
        for (int i = 0; i < 8; i++) {

            int nx = p.x + dx[i];
            int ny = p.y + dy[i];

            // 判断边界
            if (!isValid(nx, ny))
                continue;

            // 障碍物不能通过
            if (mapData.grid[ny][nx] == 1)
                continue;

            // 已经关闭的节点跳过
            if (closed[ny][nx])
                continue;

            // 判断是否为斜向移动
            bool diagonal =
                (dx[i] != 0 && dy[i] != 0);

            // 防止斜向穿过障碍物的角落
            if (diagonal) {

                // 水平方向相邻格
                int sideX = p.x + dx[i];
                int sideY = p.y;

                // 垂直方向相邻格
                int side2X = p.x;
                int side2Y = p.y + dy[i];

                if (mapData.grid[sideY][sideX] == 1 ||
                    mapData.grid[side2Y][side2X] == 1) {

                    continue;
                }
            }

            //计算移动代价 条件 ? 条件为真时的值 : 条件为假时的值
            int moveCost = diagonal
                ? DIAGONAL_COST
                : STRAIGHT_COST;

            int newG =
                gScore[p.y][p.x] + moveCost;

            // 更新更优路径
            if (newG < gScore[ny][nx]) {

                gScore[ny][nx] = newG;

                parent[ny][nx] = p;

                int h = heuristic(
                    Point(nx, ny),
                    goal
                );

                openList.push(
                    AStarNode(
                        Point(nx, ny),
                        newG,
                        h
                    )
                );
            }
        }
    }

    // ==================== 路径重建 ====================

    if (found) {

        Point current = goal;

        while (current != start) {
            //push_back() 的作用是向 vector 末尾添加元素
            mapData.path.push_back(current);

            current = parent[current.y][current.x];
        }

        mapData.path.push_back(start);

        reverse(
            mapData.path.begin(),
            mapData.path.end()
        );

        cout << "\n========== A* 搜索结果 ==========" << endl;

        cout << "搜索成功！" << endl;

        cout << "路径节点数量："
             << mapData.path.size() << endl;

        cout << "路径总代价："
             << gScore[goal.y][goal.x] << endl;

        cout << "路径长度："
             << gScore[goal.y][goal.x] / 10.0
             << " 个单位" << endl;

        cout << "搜索节点数量："
             << mapData.explored.size() << endl;

        cout << "=================================" << endl;

    } else {

        cout << "没有找到可行路径！" << endl;
    }
}

// ==================== 绘制地图 ====================

void drawMap(const MapData& mapData) {

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
                    Rect(
                        x * CELL_SIZE + 1,
                        y * CELL_SIZE + 1,
                        CELL_SIZE - 2,
                        CELL_SIZE - 2
                    ),
                    Scalar(30, 30, 30),
                    FILLED
                );
            }
        }
    }

    // 绘制已搜索区域
    for (const Point& p : mapData.explored) {

        rectangle(
            image,
            Rect(
                p.x * CELL_SIZE + 2,
                p.y * CELL_SIZE + 2,
                CELL_SIZE - 4,
                CELL_SIZE - 4
            ),
            Scalar(220, 220, 180),
            FILLED
        );
    }

    // 绘制最终路径
    for (const Point& p : mapData.path) {

        rectangle(
            image,
            Rect(
                p.x * CELL_SIZE + 2,
                p.y * CELL_SIZE + 2,
                CELL_SIZE - 4,
                CELL_SIZE - 4
            ),
            Scalar(0, 220, 255),
            FILLED
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
            CELL_SIZE / 3,
            Scalar(0, 200, 0),
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
            CELL_SIZE / 3,
            Scalar(0, 0, 255),
            FILLED
        );
    }

    // 绘制网格线
    for (int x = 0; x <= COLS; x++) {

        line(
            image,
            Point(x * CELL_SIZE, 0),
            Point(x * CELL_SIZE, WINDOW_HEIGHT),
            Scalar(200, 200, 200),
            1
        );
    }

    for (int y = 0; y <= ROWS; y++) {

        line(
            image,
            Point(0, y * CELL_SIZE),
            Point(WINDOW_WIDTH, y * CELL_SIZE),
            Scalar(200, 200, 200),
            1
        );
    }

    imshow("A* Path Planning", image);
}

// ==================== 重置地图 ====================

void resetMap(MapData& mapData) {

    mapData.grid.assign(
        ROWS,
        vector<int>(COLS, 0)
    );

    mapData.hasStart = false;
    mapData.hasGoal = false;

    mapData.path.clear();
    mapData.explored.clear();

    cout << "地图已重置，请重新设置起点和终点！"
         << endl;

    drawMap(mapData);
}

// ==================== 鼠标回调 ====================

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

    int gridX = x / CELL_SIZE;
    int gridY = y / CELL_SIZE;

    Point p(gridX, gridY);

    // 左键点击
    if (event == EVENT_LBUTTONDOWN) {

        mapData->path.clear();
        mapData->explored.clear();

        // 第一次点击设置起点
        if (!mapData->hasStart) {

            mapData->start = p;
            mapData->hasStart = true;

            cout << "起点：("
                 << gridX << ", "
                 << gridY << ")" << endl;
        }

        // 第二次点击设置终点
        else if (!mapData->hasGoal) {

            if (p == mapData->start) {

                cout << "终点不能与起点重合！"
                     << endl;

                return;
            }

            mapData->goal = p;
            mapData->hasGoal = true;

            cout << "终点：("
                 << gridX << ", "
                 << gridY << ")" << endl;
        }

        // 后续点击设置障碍物
        else {

            if (p != mapData->start &&
                p != mapData->goal) {

                mapData->grid[gridY][gridX] = 1;
            }
        }
    }

    // 右键删除障碍物
    if (event == EVENT_RBUTTONDOWN) {

        if (p != mapData->start &&
            p != mapData->goal) {

            mapData->grid[gridY][gridX] = 0;
        }

        mapData->path.clear();
        mapData->explored.clear();
    }

    drawMap(*mapData);
}

// ==================== 主函数 ====================

int main() {

    MapData mapData;

    namedWindow(
        "A* Path Planning",
        WINDOW_AUTOSIZE
    );

    setMouseCallback(
        "A* Path Planning",
        mouseCallback,
        &mapData
    );

    drawMap(mapData);

    cout << "========== A* 路径规划 ==========" << endl;

    cout << "鼠标左键第一次点击：设置起点" << endl;
    cout << "鼠标左键第二次点击：设置终点" << endl;
    cout << "鼠标左键后续点击：添加障碍物" << endl;
    cout << "鼠标右键点击：删除障碍物" << endl;

    cout << "空格键：开始搜索" << endl;
    cout << "R键：重置地图" << endl;
    cout << "ESC键：退出程序" << endl;

    cout << "=================================" << endl;

    while (true) {

        int key = waitKey(20);

        // ESC退出
         if (key == 27) {
            break;
        }

        // 空格开始搜索
        if (key == 32) {

            AStar(mapData);

            drawMap(mapData);
        }

        // R键重置
        if (key == 'r' || key == 'R') {

            resetMap(mapData);
        }
    }

    destroyAllWindows();

    return 0;
}
