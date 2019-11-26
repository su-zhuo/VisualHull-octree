#pragma warning(disable:4819)
#pragma warning(disable:4244)
#pragma warning(disable:4267)
//为尽可能提升效率，函数均写在此文件中
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <limits>
#include <vector>
#include<queue>
#include <ppl.h>
//#include<list>
// 用于判断投影是否在visual hull内部
struct Projection
{
	Eigen::Matrix<float, 3, 4> m_projMat;
	cv::Mat m_image;
	const uint m_threshold = 125;
	
	bool outOfRange(int x, int max)
	{
		return x < 0 || x >= max;
	}

	bool checkRange(float x, float y, float z)
	{
		Eigen::Vector3f vec3 = m_projMat * Eigen::Vector4f(x, y, z, 1);
		int indX = vec3[1] / vec3[2];
		int indY = vec3[0] / vec3[2];

		if (outOfRange(indX, m_image.size().height) || outOfRange(indY, m_image.size().width))
			return false;
		return m_image.at<uchar>((uint)(vec3[1] / vec3[2]), (uint)(vec3[0] / vec3[2])) > m_threshold;
	}
	
};

// 用于index和实际坐标之间的转换
struct CoordinateInfo
{
	int m_resolution;
	float m_min;
	float m_max;

	float index2coor(int index)
	{
		return m_min + index * (m_max - m_min) / m_resolution;
	}

	CoordinateInfo(int resolution = 10, float min = 0.0, float max = 100.0)
		: m_resolution(resolution)
		, m_min(min)
		, m_max(max)
	{
	}
};

#define index(i,j,k) i+2*(j)+4*(k)
enum type { root, leaf };
//八叉树数据结构
struct node
{
	int x[2];
	int y[2];
	int z[2];
	type Type;
	long num;
	node* prt;
	std::vector<node*>* child;
	bool isFull() { return (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0]) == num; }
	node(int Xmax = 100, int Ymax = 100, int Zmax = 100, int Xmin = 0, int Ymin = 0, int Zmin = 0, type _type = root)
	{
		x[1] = Xmax; x[0] = Xmin; y[1] = Ymax; y[0] = Ymin; z[1] = Zmax; z[0] = Zmin; Type = _type;
		child = new std::vector<node*>(8, nullptr);
		num = 0;
		prt = nullptr;
	}
};

class Model
{
public:
	typedef std::vector<std::vector<bool>> Pixel;	
	typedef std::vector<Pixel> Voxel;

	Model(int resX = 100, int resY = 100, int resZ = 100);
	~Model();

	void saveModel(const char* pFileName);//除测试“无法向输出速度”外用不到，故不进行提速修改，亦不调用
	void saveModelWithNormal();//存储方式改进函数，不需要形参，存储位置默认在工程\VisualHull\VisualHull中
	void loadMatrix(const char* pFileName);//用时很小，无所谓优化
	void loadImage(const char* pDir, const char* pPrefix, const char* pSuffix);//改为并行提速
	void getModel();//并行计算+八叉树模型建立
	void getSurface();//应用八叉树模型遍历
	Eigen::Vector3f getNormal(int indX, int indY, int indZ);//计算法向量，修改邻域大小可以改变速度，但是影响模型精度（肉眼不易区分）	
	void buildOctree(int x,int y,int z);//通过三维模型内（表面点）建树
	void buildSurOctree(int x, int y, int z);//通过三维模型表面点 建树
	int depth,max;
private:
	CoordinateInfo m_corrX;//实际坐标信息，用以配合相机判断点是否在三维模型内
	CoordinateInfo m_corrY;
	CoordinateInfo m_corrZ;

	int m_neiborSize;

	std::vector<Projection> m_projectionList;

	Voxel m_voxel,m_surface;	
	node* head;
	node* surHead;
};
//用于建立立体模型八叉树结构
void Model::buildOctree(int x,int y,int z) {
	//最开始深度定义为0
	int Depth = 0;	
	//定义初始的节点为头结点
	node* tNode1 = head,*tNode2;
	tNode1->num++;
	//局部变量若干
	int i, j, k, xm, ym, zm, indexN, x0, x1, y0, y1, z0, z1;
	//不断循环直至所要精度
	while (Depth != depth) {
		x0 = tNode1->x[0]; x1 = tNode1->x[1];
		y0 = tNode1->y[0]; y1 = tNode1->y[1];
		z0 = tNode1->z[0]; z1 = tNode1->z[1];
		//中心
		xm = (x0 + x1) / 2;
		ym = (y0 + y1) / 2;
		zm = (z0 + z1) / 2;
		//八个孩子中的索引，将小正方形分成八块，由上到下、左到右编号0,1,2...
		//程序模拟数学进位
		i = x < xm ? 0 : 1;
		j = y < ym ? 0 : 1;
		k = z < zm ? 0 : 1;
		indexN = index(i, j, k);
		//得到子树
		switch(indexN) {
			//八块，对当前点进行空间分配
		case 0: {
			tNode2 = (*(tNode1->child))[indexN]; 
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, ym, zm, x0, y0, z0); 
				(*(tNode1->child))[indexN]= tNode2;//将新节点放入父节点的child当中，此时，tnode2相当于父节点的子节点
				tNode2->prt = tNode1;//将子节点的父索引（PARENT）设置为tNode1
			}
			break;
		}
		case 1: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, ym, zm, xm, y0, z0);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 2: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, y1, zm, x0, ym, z0);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 3: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, y1, zm, xm, ym, z0);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 4: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, ym,z1, x0, y0, zm);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 5: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, ym,z1, xm, y0, zm);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 6: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, y1, z1, x0, ym, zm);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 7: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node( x1, y1, z1,xm, ym, zm);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		default:break;}
		tNode2->num++;//刷新当前节点的索引
		tNode1 = tNode2;//刷新父节点
		Depth++;//深度增加
	}
	tNode1->Type = leaf;//叶子节点
}
//用于建立表面明星八叉树结构
void Model::buildSurOctree(int x, int y, int z)
{
	//最开始深度定义为0
	int Depth = 0;
	//定义初始的节点为头结点
	node* tNode1 = surHead, *tNode2;
	tNode1->num++;
	//局部变量若干
	int i, j, k, xm, ym, zm, indexN, x0, x1, y0, y1, z0, z1;
	//不断循环直至所要精度
	while (Depth != depth) {
		x0 = tNode1->x[0]; x1 = tNode1->x[1];
		y0 = tNode1->y[0]; y1 = tNode1->y[1];
		z0 = tNode1->z[0]; z1 = tNode1->z[1];
		//中心
		xm = (x0 + x1) / 2;
		ym = (y0 + y1) / 2;
		zm = (z0 + z1) / 2;
		//八个孩子中的索引，将小正方形分成八块，由上到下、左到右编号0,1,2...
		//程序模拟数学进位
		i = x < xm ? 0 : 1;
		j = y < ym ? 0 : 1;
		k = z < zm ? 0 : 1;
		indexN = index(i, j, k);
		//得到子树
		switch (indexN) {
			//八块，对当前点进行空间分配
		case 0: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, ym, zm, x0, y0, z0);
				(*(tNode1->child))[indexN] = tNode2;//将新节点放入父节点的child当中，此时，tnode2相当于父节点的子节点
				tNode2->prt = tNode1;//将子节点的父索引（PARENT）设置为tNode1
			}
			break;
		}
		case 1: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, ym, zm, xm, y0, z0);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 2: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, y1, zm, x0, ym, z0);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 3: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, y1, zm, xm, ym, z0);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 4: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, ym, z1, x0, y0, zm);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 5: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, ym, z1, xm, y0, zm);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 6: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, y1, z1, x0, ym, zm);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 7: {
			tNode2 = (*(tNode1->child))[indexN];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, y1, z1, xm, ym, zm);
				(*(tNode1->child))[indexN] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		default:break;
		}
		tNode2->num++;//刷新当前节点的索引
		tNode1 = tNode2;//刷新父节点
		Depth++;//深度增加
	}
	tNode1->Type = leaf;//叶子节点
}

Model::Model(int resX, int resY, int resZ)
	: m_corrX(resX, -5, 5)
	, m_corrY(resY, -10, 10)
	, m_corrZ(resZ, 15, 30)
{
	if (resX > 100)
		m_neiborSize = resX / 100;
	else
		m_neiborSize = 1;
	m_voxel = Voxel(m_corrX.m_resolution, Pixel(m_corrY.m_resolution, std::vector<bool>(m_corrZ.m_resolution, true)));
	m_surface = Voxel(m_corrX.m_resolution, Pixel(m_corrY.m_resolution, std::vector<bool>(m_corrZ.m_resolution, false)));

	depth = 0;
	max = 1;
	while (max < resX || max < resY || max < resZ) { 
		depth++; max *= 2;
	}
	head = new node(max, max, max);//初始化根节点
	surHead = new node(max, max, max);//初始化根节点
}
Model::~Model()
{
}

//因为仅存储模型，不带法向量做存储和saveModelWithNormal()功能重复，故可以不用，不做修改
void Model::saveModel(const char* pFileName)
{
	std::ofstream fout(pFileName);

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
				if (m_surface[indexX][indexY][indexZ])
				{
					float coorX = m_corrX.index2coor(indexX);
					float coorY = m_corrY.index2coor(indexY);
					float coorZ = m_corrZ.index2coor(indexZ);
					fout << coorX << ' ' << coorY << ' ' << coorZ << std::endl;
				}
}
//提速修改：广搜对surface的八叉树进行遍历，修改后fprintf函数写文件位置在\VisualHull\VisualHull中，注意.xzy转.ply位置
void Model::saveModelWithNormal()
{
	std::FILE *FSPOINTER;	
	FSPOINTER = fopen("WithNormal.xyz", "w");//使用函数fprintf可以有效提高写.xyz速度
	
	std::queue<node*> q;//队列元素 为节点指针
	q.push(surHead);//把head接到队列末端
	node* tNode;
	while (q.size() != 0) {//队列内含元素多少
		tNode = q.front();//返回第一个元素
		q.pop();//弹出队列的第一个元素
		if (tNode->Type == leaf) {
			float coorX = m_corrX.index2coor(tNode->x[0]);
			float coorY = m_corrY.index2coor(tNode->y[0]);
			float coorZ = m_corrZ.index2coor(tNode->z[0]);
			fprintf(FSPOINTER, "%f %f %f ", coorX, coorY, coorZ);//坐标信息，注意格式、没有换行
			Eigen::Vector3f nor = getNormal(tNode->x[0], tNode->y[0], tNode->z[0]);
			fprintf(FSPOINTER, "%f %f %f%c", nor(0), nor(1), nor(2), '\n');//法向量信息，注意格式、有换行
		}
		else {
			if (tNode->num != 0 && !tNode->isFull()) {
				for (int i = 0; i < 8; i++) {
					if ((*(tNode->child))[i] != nullptr)
						q.push((*(tNode->child))[i]);
				}
			}
		}
	}
}
//未做修改：实验用时0.002s
void Model::loadMatrix(const char* pFileName)
{
	std::ifstream fin(pFileName);

	int num;
	Eigen::Matrix<float, 3, 3> matInt;
	Eigen::Matrix<float, 3, 4> matExt;
	Projection projection;
	while (fin >> num)
	{
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				fin >> matInt(i, j);

		float temp;
		fin >> temp;
		fin >> temp;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++)
				fin >> matExt(i, j);

		projection.m_projMat = matInt * matExt;
		m_projectionList.push_back(projection);
	}
}
//提速修改：并行计算，进行快速读图：20张图 约从0.166s降到约0.091s
void Model::loadImage(const char* pDir, const char* pPrefix, const char* pSuffix)
{
	int fileCount = m_projectionList.size();
	std::string fileName(pDir);
	fileName += '/';
	fileName += pPrefix;
	//for (int i = 0; i < fileCount; i++)
	concurrency::parallel_for(0, fileCount, [&](int i)
	{
		std::cout << fileName + std::to_string(i) + pSuffix << std::endl;
		m_projectionList[i].m_image = cv::imread(fileName + std::to_string(i) + pSuffix, CV_8UC1);//usigned8bits，1通道图像
	});
}
//提速修改：并行遍历，提前停止，建立立方体的八叉树结构
void Model::getModel()//必须完全遍历
{
	int prejectionCount = m_projectionList.size();

	//for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
	concurrency::parallel_for(0, m_corrX.m_resolution, [&](size_t indexX)
	{
		//for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
		concurrency::parallel_for(0, m_corrY.m_resolution, [&](size_t indexY)
		{
			//for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
			concurrency::parallel_for(0, m_corrZ.m_resolution, [&](size_t indexZ)
			{
				//parallel_for(0, prejectionCount, [&](size_t i)
				for (int i = 0; i < prejectionCount; i++)
				{
					float coorX = m_corrX.index2coor(indexX);
					float coorY = m_corrY.index2coor(indexY);
					float coorZ = m_corrZ.index2coor(indexZ);
					if (!(m_voxel[indexX][indexY][indexZ] = m_projectionList[i].checkRange(coorX, coorY, coorZ)))break;

				}
				if (m_voxel[indexX][indexY][indexZ]) buildOctree(indexX, indexY, indexZ);

			});
		});
	});
}
//提速修改：遍历立方体，得到表面模型，建立表面的八叉树结构
void Model::getSurface()
{
	// 邻域：上、下、左、右、前、后。
	int dx[6] = { -1, 0, 0, 0, 0, 1 };
	int dy[6] = { 0, 1, -1, 0, 0, 0 };
	int dz[6] = { 0, 0, 0, 1, -1, 0 };

	auto outOfRange = [&](int indexX, int indexY, int indexZ){
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};//超出即true

	std::queue<node*> q;//队列元素为八叉树节点指针
	q.push(head);//把head接到队列末端
	node* tNode;
	while (q.size() != 0) {
		tNode = q.front();//返回第一个
		q.pop();//弹出
		if (tNode->Type == leaf) {
			bool ans = false;
			for (int i = 0; i < 6; i++)
			{
				ans = ans ||outOfRange(tNode->x[0] + dx[i], tNode->y[0] + dy[i], tNode->z[0] + dz[i])
					|| !m_voxel[tNode->x[0] + dx[i]][tNode->y[0] + dy[i]][tNode->z[0] + dz[i]];
				if (ans){ //基本思想就是只要邻域有零即ans为真，该点为表面点
					m_surface[tNode->x[0]][tNode->y[0]][tNode->z[0]]=true; buildSurOctree(tNode->x[0],tNode->y[0],tNode->z[0]);//可设置变量自加计算表面节点数				
					break;
				}
			}
		}
		else 
		{
			if (tNode->num != 0 && !tNode->isFull()) {
				for (int i = 0; i < 8; i++) {
					if ((*(tNode->child))[i] != nullptr)
						q.push((*(tNode->child))[i]);
				}
			}
		}		
	}
}
//未做修改（PCA主成分分析）可通过邻域大小修改改变计算速度，但是精度会发生变化
Eigen::Vector3f Model::getNormal(int indX, int indY, int indZ)
{
	auto outOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};

	std::vector<Eigen::Vector3f> neiborList;//
	std::vector<Eigen::Vector3f> innerList;//
	//可以不找立体邻域么？找表面上的邻域（以此简化程序？）
	//m_neiborSize = 2;//此处可以调整，会影响速率
	for (int dX = -m_neiborSize; dX <= m_neiborSize; dX++)//std::cout << m_neiborSize << endl;输出为3
		//parallel_for(-m_neiborSize, m_neiborSize, [&](int dX)//矩阵运算，尝试并行无效。
		for (int dY = -m_neiborSize; dY <= m_neiborSize; dY++)
			for (int dZ = -m_neiborSize; dZ <= m_neiborSize; dZ++)
			{
				if (!dX && !dY && !dZ)
					continue;
				int neiborX = indX + dX;
				int neiborY = indY + dY;
				int neiborZ = indZ + dZ;
				if (!outOfRange(neiborX, neiborY, neiborZ))//超出为true 去反为false
				{
					float coorX = m_corrX.index2coor(neiborX);
					float coorY = m_corrY.index2coor(neiborY);
					float coorZ = m_corrZ.index2coor(neiborZ);
					if (m_surface[neiborX][neiborY][neiborZ])
						neiborList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));//vector尾部加一个数据
					else if (m_voxel[neiborX][neiborY][neiborZ])
						innerList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
				}
			}

	Eigen::Vector3f point(m_corrX.index2coor(indX), m_corrY.index2coor(indY), m_corrZ.index2coor(indZ));//point?

	Eigen::MatrixXf matA(3, neiborList.size());//矩阵类型变量
	for (int i = 0; i < neiborList.size(); i++)
		matA.col(i) = neiborList[i] - point;//为什么减这个point
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA * matA.transpose());
	Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
	int indexEigen = 0;
	if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
		indexEigen = 1;
	if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
		indexEigen = 2;
	Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);

	Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
	for (auto const& vec : innerList)
		innerCenter += vec;
	innerCenter /= innerList.size();

	if (normalVector.dot(point - innerCenter) < 0)
		normalVector *= -1;
	return normalVector;
}
//省去无用步骤，逐步计时
int main(int argc, char** argv)
{
	clock_t t = clock();//开始计时
	clock_t t1, t2, t3, t4, t5, t6, t7, t8;
	// 分别设置xyz方向的Voxel（三维像素）分辨率
	Model model(300, 300, 300);
	t1 = clock() - t;
	std::cout << "time: " << (float(t1) / CLOCKS_PER_SEC) << "seconds\n";

	// 读取相机的内外参数
	model.loadMatrix("../../calibParamsI.txt");
	t2 = clock() - t - t1;
	std::cout << "time: " << (float(t2) / CLOCKS_PER_SEC) << "seconds\n";

	// 读取投影图片
	model.loadImage("../../wd_segmented", "WD2_", "_00020_segmented.png");
	t3 = clock() - t - t1 - t2;
	std::cout << "time: " << (float(t3) / CLOCKS_PER_SEC) << "seconds\n";

	// 得到Voxel模型
	model.getModel();
	std::cout << "get model done\n";
	t4 = clock() - t - t1 - t2 - t3;
	std::cout << "time: " << (float(t4) / CLOCKS_PER_SEC) << "seconds\n";

	// 获得Voxel模型的表面
	model.getSurface();
	std::cout << "get surface done\n";
	t5 = clock() - t - t1 - t2 - t3 - t4;
	std::cout << "time: " << (float(t5) / CLOCKS_PER_SEC) << "seconds\n";

	// 将模型导出为xyz格式，和下面的函数功能重复
	//model.saveModel("../../WithoutNormal.xyz");
	//std::cout << "save without normal done\n";
	t6 = clock() - t - t1 - t2 - t3 - t4 - t5;
	//std::cout << "time: " << (float(t6) / CLOCKS_PER_SEC) << "seconds\n";

	// 获得代法向量信息的.xyz文件
	model.saveModelWithNormal();
	std::cout << "save with normal done\n";
	t7 = clock() - t - t1 - t2 - t3 - t4 - t5 - t6;
	std::cout << "time: " << (float(t7) / CLOCKS_PER_SEC) << "seconds\n";

	//泊松表面重构
	system("PoissonRecon.x64 --in WithNormal.xyz --out ../../mesh.ply");
	std::cout << "save mesh.ply done\n";
	t8 = clock() - t - t1 - t2 - t3 - t4 - t5 - t6 - t7;
	std::cout << "possonRecon_tim: " << (float(t8) / CLOCKS_PER_SEC) << "seconds\n";
	t = clock() - t;
	std::cout << "total_time: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";
	std::cout << "total_time-possonRecon_time: " << (float(t- t8) / CLOCKS_PER_SEC) << "seconds\n";
	system("pause");
	return (0);
}