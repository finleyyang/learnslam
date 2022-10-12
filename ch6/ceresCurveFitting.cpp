//
// Created by xiang on 18-11-19.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// 代价函数的计算模型
struct CURVE_FITTING_COST {
  CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

  // 残差的计算
  template<typename T>
  bool operator()(
    const T *const abc, // 模型参数，有3维
    T *residual) const {
    residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y-exp(ax^2+bx+c)
    return true;
  }

  const double _x, _y;    // x,y数据
};

int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
  double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
  int N = 100;                                 // 数据点
  double w_sigma = 1.0;                        // 噪声Sigma值
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;                                 // OpenCV随机数产生器

  vector<double> x_data, y_data;      // 数据
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
  }

  double abc[3] = {ae, be, ce};

  // 构建最小二乘问题
  ceres::Problem problem;
  for (int i = 0; i < N; i++) {
      /* 第一个参数 CostFunction* : 描述最小二乘的基本形式即代价函数 例如书上的137页fi(.)的形式
        * 第二个参数 LossFunction* : 描述核函数的形式 例如书上的ρi（.）
          * 第三个参数 double* :       待估计参数（用数组存储）
          * 这里仅仅重载了三个参数的函数，如果上面的double abc[3]改为三个double a=0 ,b=0,c = 0;
53          * 此时AddResidualBlock函数的参数除了前面的CostFunction LossFunction 外后面就必须加上三个参数 分别输入&a,&b,&c
54          * 那么此时下面的 ceres::AutoDiffCostFunction<>模板参数就变为了 <CURVE_FITTING_COST,1,1,1,1>后面三个1代表有几类未知参数
55          * 我们修改为了a b c三个变量，所以这里代表了3类，之后需要在自己写的CURVE_FITTING_COST类中的operator()函数中，
56          * 把形式参数变为了 const T* const a, const T* const b, const T* const c ,T* residual
57          * 上面修改的方法与本例程实际上一样，只不过修改的这种方式显得乱，实际上我们在用的时候，一般都是残差种类有几个，那么后面的分类 就分几类
58          * 比如后面讲的重投影误差，此事就分两类 一类是相机9维变量，一类是点的3维变量，然而残差项变为了2维
59          *
60          * （1）: 修改后的写法（当然自己定义的代价函数要对应修改重载函数的形式参数，对应修改内部的残差的计算）：
61          *      ceres::CostFunction* cost_function
62          *              = new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 1 ,1 ,1>(
63          *                  new CURVE_FITTING_COST ( x_data[i], y_data[i] ) );
64          *      problem.AddResidualBlock(cost_function,nullptr,&a,&b,&c);
65          * 修改后的代价函数的计算模型:
66          *   struct CURVE_FITTING_COST
67          *   {
68          *       CURVE_FITTING_COST ( double x, double y ) : _x ( x ), _y ( y ) {}
69          *       // 残差的计算
70          *       template <typename T>
71          *       bool operator() (
72          *          const T* const a,
73          *          const T* const b,
74          *          const T* const c,
75          *          T* residual   ) const     // 残差
76          *       {
77          *           residual[0] = T ( _y ) - ceres::exp ( a[0]*T ( _x ) *T ( _x ) + b[0]*T ( _x ) + c[0] ); // y-exp(ax^2+bx+c)
78          *           return true;
79          *       }
80          *       const double _x, _y;    // x,y数据
81          *   };//代价类结束
82          *
83          *
84          * （2）: 本例程下面的语句通常拆开来写(看起来方便些):
85          * ceres::CostFunction* cost_function
86          *              = new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
87          *                  new CURVE_FITTING_COST ( x_data[i], y_data[i] ) );
88          * problem.AddResidualBlock(cost_function,nullptr,abc)
89          * */
      // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
      new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
        new CURVE_FITTING_COST(x_data[i], y_data[i])
      ),
      nullptr,            // 核函数，这里不使用，为空
      abc                 // 待估计参数
    );
  }

  // 配置求解器
  ceres::Solver::Options options;     // 这里有很多配置项可以填
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 增量方程如何求解
  options.minimizer_progress_to_stdout = true;   // 输出到cout

  ceres::Solver::Summary summary;                // 优化信息
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);  // 开始优化
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  // 输出结果
  cout << summary.BriefReport() << endl;
  cout << "estimated a,b,c = ";
  for (auto a:abc) cout << a << " ";
  cout << endl;

  return 0;
}