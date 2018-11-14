#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <omp.h>

#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))

//graph500
#define Efactor 16
#define A     0.57
#define B     0.19
#define C     0.19

#define WORD    6          //64 bits, 2^6
#define MASK    0x0000003F //6 1's
#define MILLION 1000000L

uint32_t verbose = 0;

uint32_t numT;       //number of threads
uint32_t *prefixSum; //offsets for threads to write in sparse

uint64_t seed;               //first seed for pseudorandom number generators
#define GAP 16807            //7^5 between each subsequent seed
struct drand48_data *rndBuf; //one buffer per thread

//graph data structure
uint32_t scale;                  //graph500
uint32_t n, m, bitN;             //num vertices, num edges, bitN = n >> WORD
uint32_t *csr, *idxCSR, *tmpCSR; //compressed sparse row
uint32_t *edgeU, *edgeV, *deg, **neighbor, *idx, *newID;
uint64_t *hubs; //adjacency to top 64 hub vertices

//BFS data structure
enum direction {
  TopDown1, BottomUp, BU2TD, TopDown2
};
uint32_t layerSize, layerDeg;
uint32_t *sparse, sparseSize, *newSparse, newSparseSize;
uint32_t **mySparse, *mySparseSize;
uint64_t *dense, *newDense;
int *parent;        //-1 if unvisited
uint32_t root;      //current root of BFS
uint32_t roots[64]; //64 randomly selected roots for BFS

//statistics for 64 runs of BFS
float cm[64], runTime[64], teps[64], statistics[7];

uint64_t oneBit[64];
//oneBit[0] 0001
//oneBit[1] 0010
//oneBit[2] 0100
//oneBit[3] 1000

/* https://graph500.org/?page_id=12#sec-3 */
void generate(void) {
  uint32_t i, j, shift, bit, tid;
  double ab, aNorm, cNorm, rnd;

  ab = A + B;
  aNorm = A / ab;
  cNorm = C / (1 - ab);
  for (i = 0; i < m; i++)
    edgeU[i] = edgeV[i] = 0; //zero-based indexing

#pragma omp parallel for private(i,shift,tid,bit,rnd)
  for (j = 0; j < m; j++) {
    tid = omp_get_thread_num();
    for (i = 0; i < scale; i++) {
      shift = (i == 0) ? 1 : (1 << i);
      drand48_r(&rndBuf[tid], &rnd);
      bit = rnd > ab;
      edgeU[j] += bit * shift;
      drand48_r(&rndBuf[tid], &rnd);
      bit = rnd > (bit ? cNorm : aNorm);
      edgeV[j] += bit * shift;
    }
  }
}

void init(void) {
  uint32_t i;

  n = 1 << scale;
  m = n * Efactor;
  bitN = n >> WORD;

  rndBuf = NULL;
  parent = NULL;
  neighbor = NULL;
  mySparse = NULL;
  dense = newDense = NULL;
  sparse = newSparse = mySparseSize = NULL;
  deg = edgeU = edgeV = idx = newID = tmpCSR = csr = idxCSR = NULL;

  posix_memalign((void **)&parent,       64, sizeof(int) * n);
  posix_memalign((void **)&csr,          64, sizeof(uint32_t) * m * 2);
  posix_memalign((void **)&idxCSR,       64, sizeof(uint32_t) * (n + 1));
  posix_memalign((void **)&dense,        64, n >> 3); // n/8 bytes
  posix_memalign((void **)&newDense,     64, n >> 3); // n/8 bytes
  posix_memalign((void **)&sparse,       64, sizeof(uint32_t) * n);
  posix_memalign((void **)&newSparse,    64, sizeof(uint32_t) * n);
  posix_memalign((void **)&mySparse,     64, sizeof(uint32_t*) * numT);
  posix_memalign((void **)&mySparseSize, 64, sizeof(uint32_t) * numT);
  posix_memalign((void **)&prefixSum,    64, sizeof(uint32_t) * (numT + 1));

  if (parent == NULL || csr == NULL || idxCSR == NULL || dense == NULL ||
      newDense == NULL || sparse == NULL || newSparse == NULL ||
      mySparse == NULL || mySparseSize == NULL || prefixSum == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(0);
  }
  for (i = 0; i < numT; i++) {
    posix_memalign((void **) &mySparse[i], 64, sizeof(uint32_t) * (n >> 4));
    if (mySparse[i] == NULL) {
      fprintf(stderr, "out of memory\n");
      exit(0);
    }
  }

  posix_memalign((void **)&rndBuf,   64, sizeof(struct drand48_data) * numT);
  posix_memalign((void **)&tmpCSR,   64, sizeof(uint32_t) * m * 2);
  posix_memalign((void **)&deg,      64, sizeof(uint32_t) * n);
  posix_memalign((void **)&idx,      64, sizeof(uint32_t) * n);
  posix_memalign((void **)&newID,    64, sizeof(uint32_t) * n);
  posix_memalign((void **)&neighbor, 64, sizeof(uint32_t*) * n);
  posix_memalign((void **)&edgeU,    64, sizeof(uint32_t) * m);
  posix_memalign((void **)&edgeV,    64, sizeof(uint32_t) * m);

  if (rndBuf == NULL || tmpCSR == NULL || deg == NULL || idx == NULL ||
      newID == NULL || neighbor == NULL || edgeU == NULL || edgeV == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(0);
  }
  srand48_r(seed, &rndBuf[0]);
  for (i = 1; i < numT; i++)
    srand48_r(seed + i * GAP, &rndBuf[i]);
  oneBit[0] = 1;
  for (i = 1; i < 64; i++)
    oneBit[i] = oneBit[i-1] << 1;
}

//ascending
int cmpID(const void *a, const void *b) {
  return *(uint32_t*) a - *(uint32_t*) b;
}

//descending
int cmpDeg(const void *a, const void *b) {
  return deg[*(uint32_t*) b] - deg[*(uint32_t*) a];
}

//descending degree, break tie by having a higher degree neighbor
int degBrkTie(const void *a, const void *b) {
  uint32_t u, v;

  u = *(uint32_t*) a;
  v = *(uint32_t*) b;
  if (deg[u] > deg[v])
    return -1;
  if (deg[u] < deg[v])
    return 1;
  if (hubs[u] > hubs[v])
    return -1;
  if (hubs[u] < hubs[v])
    return 1;
  return 0;
}

//ascending
int cmpNewID(const void *a, const void *b) {
  return newID[*(uint32_t*) a] - newID[*(uint32_t*) b];
}

void preprocessing(void) {
  uint32_t i, j, k, u, v, chunk, start, end;
  double rnd;

  chunk = m / numT;

#pragma omp parallel for private(j,u,v,start,end)
  for (i = 0; i < numT; i++) {
    start = i * chunk;
    end = (i == numT - 1) ? m : (start + chunk);
    for (j = 0; j < m; j++) {
      u = edgeU[j], v = edgeV[j];
      if (start <= u && u < end)
	deg[u]++;
      if (start <= v && v < end)
	deg[v]++;
    }
  }

  idxCSR[0] = 0;
  for (i = 0; i < n; i++) {
    neighbor[i] = &tmpCSR[ idxCSR[i] ];
    idxCSR[i + 1] = idxCSR[i] + deg[i];
    deg[i] = 0;
    idx[i] = i;
  }

#pragma omp parallel for private(j,u,v,start,end)
  for (i = 0; i < numT; i++) {
    start = i * chunk;
    end = (i == numT - 1) ? m : (start + chunk);
    for (j = 0; j < m; j++) {
      u = edgeU[j], v = edgeV[j];
      //symmetry
      if (start <= u && u < end)
	neighbor[u] [ deg[u]++ ] = v;
      if (start <= v && v < end)
	neighbor[v] [ deg[v]++ ] = u;
    }
  }

  free(edgeV);
  free(edgeU);

  //remove self-loops and parallel edges
#pragma omp parallel for private(j,k) schedule(guided)
  for (i = 0; i < n; i++) {
    if (deg[i] <= 1)
      continue;
    qsort((void *)neighbor[i], deg[i], sizeof(uint32_t), cmpID);
    for (j = k = 0; j < deg[i]; j++) {
      if (neighbor[i] [j] == i) //self-loop
	continue;
      if (k > 0 && neighbor[i] [j] == neighbor[i] [k - 1]) //parallel edges
	continue;
      neighbor[i] [k++] = neighbor[i] [j];
    }
    deg[i] = k;
  }

  posix_memalign((void **)&hubs, 64, sizeof(uint64_t) * n);
  for (i = 0; i < n; i++)
    hubs[i] = 0;
  //sort by nonincreasing degrees
  qsort((void *)idx, n, sizeof(uint32_t), cmpDeg);
  for (i = 0; i < 64; i++) {
    for (j = 0; j < deg[ idx[i] ]; j++) {
      u = neighbor[ idx[i] ] [j];
      hubs[u] = hubs[u] | oneBit[64 - i - 1];
    }
  }
  //sort by nonincreasing degrees, break tie
  qsort((void *)idx, n, sizeof(uint32_t), degBrkTie);

#pragma omp parallel for
  for (i = 0; i < n; i++)
    newID[ idx[i] ] = i;

  //sort adjacency lists by newIDs
#pragma omp parallel for
  for (i = 0; i < n; i++)
    if (deg[i] > 1)
      qsort((void *)neighbor[i], deg[i], sizeof(uint32_t), cmpNewID);

  idxCSR[0] = 0;
  for (i = 0; i < n; i++)
    idxCSR[i + 1] = idxCSR[i] + deg[ idx[i] ];
  m = idxCSR[n];

  //remove degree-0 vertices
  while (deg[ idx[n-1] ] == 0)
    n--;
  //round up n to the next multiple of 512
  if (n & 0x000001FF)
    n = ((n >> 9) + 1) << 9;

  for (i = 0, k = 0; i < n; i++)
    for (j = 0; j < deg[ idx[i] ]; j++)
      csr[k++] = newID[ neighbor[ idx[i] ] [j] ];

  for (i = 0; i < 64; i++) {
    //pick 64 different roots for BFS randomly
    while (( 1 )) {
      drand48_r(&rndBuf[0], &rnd);
      roots[i] = floor(rnd * n);
      if (deg[ idx[ roots[i] ] ] == 0)
	continue;
      for (j = 0; j < i; j++)
	if (roots[i] == roots[j])
	  break;
      if (i == j)
	break;
    }
  }

  free(neighbor);
  free(newID);
  free(idx);
  free(deg);
  free(tmpCSR);
  free(rndBuf);
}

void topDown(void) {
  uint32_t tid, i, j, u, v, outDeg = 0;

#pragma vector aligned

  for (i = 0; i < numT; i++)
    mySparseSize[i] = 0;

  if (layerSize == 1) {
    u = sparse[0];
#pragma omp parallel for private(tid,v) reduction(+:outDeg) schedule(guided)
    for (i = idxCSR[u]; i < idxCSR[u + 1]; i++) {
      tid = omp_get_thread_num();
      v = csr[i];
      if (parent[v] == -1) {
	parent[v] = u;
	mySparse[tid] [ mySparseSize[tid]++ ] = v;
	outDeg += idxCSR[v + 1] - idxCSR[v];
      }
    }
  }
  else { //layerSize > 1
#pragma omp parallel for private(tid,j,u,v) reduction(+:outDeg) \
  schedule(guided)
    for (i = 0; i < layerSize; i++) {
      tid = omp_get_thread_num();
      u = sparse[i];
      for (j = idxCSR[u]; j < idxCSR[u + 1]; j++) {
	v = csr[j];
	if (parent[v] == -1) {
	  //race condition, multiple copies of v may be added
	  parent[v] = u;
	  mySparse[tid] [ mySparseSize[tid]++ ] = v;
	  outDeg += idxCSR[v + 1] - idxCSR[v];
	}
      }
    }
  }

  layerDeg = outDeg;
  prefixSum[0] = 0;
  for (i = 0; i < numT; i++)
    prefixSum[i + 1] = prefixSum[i] + mySparseSize[i];

#pragma omp parallel for private(tid,j)
  for (i = 0; i < numT; i++) {
    tid = omp_get_thread_num();
    for (j = 0; j < mySparseSize[tid]; j++)
      newSparse[prefixSum[tid] + j] = mySparse[tid] [j];
  }

  layerSize = prefixSum[numT];
}

void td2bu(void) {
  uint32_t i, j, u, w, o, chunk, rem, start, end;

#pragma vector aligned
  chunk = (n >> 9) / numT;
  rem   = (n >> 9) % numT;

#pragma omp parallel for
  for (i = 0; i < bitN; i++)
    dense[i] = 0;

#pragma omp parallel for private(start,end,j,u,w,o) schedule(static,1)
  for (i = 0; i < numT; i++) {
    start = i * chunk;
    end = start + chunk;
    if (rem) {
      if (i < rem) start +=   i, end += i + 1;
      else         start += rem, end += rem;
    }
    start <<= 9, end <<= 9;
    for (j = 0; j < layerSize; j++) {
      u = sparse[j];
      if (u < start || u >= end)
	continue;
      w = u >> WORD; //64-bit word number
      o = u &  MASK; //offset within word
      dense[w] = dense[w] | oneBit[o];
    }
  }
}

void bottomUp2TopDown(void) {
  uint32_t tid, i, j, k, a, d, e, u, v;
  uint32_t *ptr32;
  uint16_t *ptr16;
  uint8_t  *ptr8, mask;

#pragma vector aligned
  for (i = 0; i < numT; i++)
    mySparseSize[i] = 0;

#pragma omp parallel for private(tid,j,k,u,v,a,d,e,ptr32,ptr16,ptr8,mask) \
  schedule(guided)
  for (i = 0; i < bitN; i++) {
    tid = omp_get_thread_num();
    if (dense[i] == 0)
      continue; //dense[i] is all zeros
    ptr32 = (uint32_t*) &dense[i];
    for (j = 0; j < 2; j++) {
      if (ptr32[j] == 0)
	continue;
      ptr16 = (uint16_t*) &ptr32[j];
      for (k = 0; k < 2; k++) {
	if (ptr16[k] == 0)
	  continue;
	ptr8 = (uint8_t*) &ptr16[k];
	for (a = 0; a < 2; a++) {
	  if (ptr8[a] == 0)
	    continue;
	  mask = ptr8[a];
	  d = 0;
	  while (mask) {
	    if (mask & 1) {
	      u = (i << 6) + (j << 5) + (k << 4) + (a << 3) + d;
	      //visit vertex u
	      for (e = idxCSR[u]; e < idxCSR[u + 1]; e++) {
		v = csr[e];
		if (parent[v] == -1) {
		  //race condition, multiple copies of v may be added
		  parent[v] = u;
		  mySparse[tid] [ mySparseSize[tid]++ ] = v;
		}
	      }
	    }
	    mask >>= 1;
	    d++;
	  }
	}
      }
    }
  }

  prefixSum[0] = 0;
  for (i = 0; i < numT; i++)
    prefixSum[i + 1] = prefixSum[i] + mySparseSize[i];

#pragma omp parallel for private(tid,j)
  for (i = 0; i < numT; i++) {
    tid = omp_get_thread_num();
    for (j = 0; j < mySparseSize[tid]; j++)
      newSparse[prefixSum[tid] + j] = mySparse[tid] [j];
  }

  layerSize = prefixSum[numT];
  layerDeg = 0; //no longer needed
}

void bottomUp(void) {
  uint32_t i, j, u, w, o, numV = 0, outDeg = 0;
  uint64_t tmp;

#pragma vector aligned
#pragma omp parallel for
  for (i = 0; i < bitN; i++)
    newDense[i] = 0;

#pragma omp parallel for private(j,u,w,o,tmp) reduction(+:numV,outDeg)	\
  schedule(dynamic,512)
  for (i = 0; i < n; i++) {
    if (parent[i] != -1)
      continue;
    for (j = idxCSR[i]; j < idxCSR[i + 1]; j++) {
      u = csr[j];    //u has an edge going into vertex i
      w = u >> WORD; //64-bit word number
      o = u &  MASK; //offset within word
      tmp = dense[w] & oneBit[o];
      //Is u in the layer?
      if (tmp == 0) //No.
	continue;
      //Add vertex i to the new layer
      parent[i] = u;
      w = i >> WORD; //64-bit word number
      o = i &  MASK; //offset within word
      newDense[w] = newDense[w] | oneBit[o];
      outDeg += idxCSR[i+1] - idxCSR[i];
      numV++;
      break;
    }
  }

  layerSize = numV;
  layerDeg = outDeg;
}

//reset BFS data structure
void reset(void) {
  uint32_t i;

  for (i = 0; i < n; i++)
    parent[i] = -1;
  parent[root] = root;

  sparse[0] = root;
  layerSize = 1;
  layerDeg = idxCSR[root+1] - idxCSR[root];
}

void bfs(void) {
  enum direction dir = TopDown1, nextDir;
  uint32_t numIter = 0, *tmpPtr;
  uint64_t *ptr64;

  while (( 1 )) {
    numIter++;
    if (verbose)
      printf("iter %u: s %u dir %u\n", numIter, layerSize, dir);

    if (dir == TopDown1 || dir == TopDown2)
      topDown();
    else if (dir == BottomUp)
      bottomUp();
    else { //dir == BU2TD
      bottomUp2TopDown();
      dir = TopDown2;
    }

    if (layerSize == 0)
      break; //we are done

    if (dir == TopDown1) {
      tmpPtr = sparse, sparse = newSparse, newSparse = tmpPtr;
      if (layerSize <= n >> 5 && layerDeg <= n >> 5)
	nextDir = TopDown1;
      else {
	nextDir = BottomUp;
	td2bu();
      }
    }
    else if (dir == BottomUp) {
      ptr64 = dense, dense = newDense, newDense = ptr64;
      if (layerSize <= n >> 5 && layerDeg <= n >> 5)
	nextDir = BU2TD;
      else
	nextDir = BottomUp;
    }
    else { //dir == TopDown2
      tmpPtr = sparse, sparse = newSparse, newSparse = tmpPtr;
      nextDir = TopDown2;
    }
    dir = nextDir;
  }
}

int cmpFloat(const void *a, const void *b) {
  if (*(float*) a < *(float*) b)
    return -1;
  if (*(float*) a > *(float*) b)
    return 1;
  return 0;
}

void calcStat(float data[]) {
  float sum, mean, std;
  uint32_t i;

  qsort((void *)data, 64, sizeof(float), cmpFloat);
  statistics[0] = data[0];
  statistics[1] = data[15];
  statistics[2] = data[31];
  statistics[3] = data[47];
  statistics[4] = data[63];
  sum = 0.0;
  for (i = 0; i < 64; i++)
    sum += data[i];
  mean = sum / 64.0;
  sum = 0.0;
  for (i = 0; i < 64; i++)
    sum += (data[i] - mean) * (data[i] - mean);
  std = sqrt(sum / 63.0);
  statistics[5] = mean;
  statistics[6] = std;
}

void output(float kernel1) {
  float sum, hMean, hSTD;
  uint32_t i;

  printf("SCALE: %u\n", scale);
  printf("NBFS: 64\n");
  printf("construction_time: %20.17e\n", kernel1);

  calcStat(runTime);
  printf("bfs_min_time: %20.17e\n", statistics[0]);
  printf("bfs_firstquartile_time: %20.17e\n", statistics[1]);
  printf("bfs_median_time: %20.17e\n", statistics[2]);
  printf("bfs_thirdquartile_time: %20.17e\n",statistics[3]);
  printf("bfs_max_time: %20.17e\n", statistics[4]);
  printf("bfs_mean_time: %20.17e\n", statistics[5]);
  printf("bfs_stddev_time: %20.17e\n", statistics[6]);

  calcStat(cm);
  printf ("bfs_min_nedge: %20.17e\n", statistics[0]);
  printf ("bfs_firstquartile_nedge: %20.17e\n", statistics[1]);
  printf ("bfs_median_nedge: %20.17e\n", statistics[2]);
  printf ("bfs_thirdquartile_nedge: %20.17e\n", statistics[3]);
  printf ("bfs_max_nedge: %20.17e\n", statistics[4]);
  printf ("bfs_mean_nedge: %20.17e\n", statistics[5]);
  printf ("bfs_stddev_nedge: %20.17e\n", statistics[6]);

  calcStat(teps);
  sum = 0.0;
  for (i = 0; i < 64; i++)
    sum += 1.0 / teps[i];
  hMean = 1.0 / (sum / 64.0);
  sum = 0.0;
  for (i = 0; i < 64; i++)
    sum += (1.0 / teps[i] - 1.0 / hMean) * (1.0 / teps[i] - 1.0 / hMean);
  hSTD = (sqrt(sum) / 63.0) * hMean * hMean;
  statistics[5] = hMean;
  statistics[6] = hSTD;
  printf ("bfs_min_TEPS: %20.17e\n", statistics[0]);
  printf ("bfs_firstquartile_TEPS: %20.17e\n", statistics[1]);
  printf ("bfs_median_TEPS: %20.17e\n", statistics[2]);
  printf ("bfs_thirdquartile_TEPS: %20.17e\n", statistics[3]);
  printf ("bfs_max_TEPS: %20.17e\n", statistics[4]);
  printf ("bfs_harmonic_mean_TEPS: %20.17e\n", statistics[5]);
  printf ("bfs_harmonic_stddev_TEPS: %20.17e\n", statistics[6]);
}

int main(int argc, char* argv[]) {
  float kernel1, sec, minimum;
  struct timeval start, stop;
  uint32_t i, j, cntM;
  int c;

  seed = 1;
  scale = 22;
  numT = omp_get_max_threads();
  while ((c = getopt(argc, argv, "d:s:t:v:")) != -1) {
    switch (c) {
    case 'd': sscanf(optarg, "%lu", &seed);   break; //seed srand48_r
    case 's': sscanf(optarg, "%u", &scale);   break; //scale, n = 2^s
    case 't': sscanf(optarg, "%u", &numT);    break;
    case 'v': sscanf(optarg, "%u", &verbose); break;
    default: break;
    }
  }
  if (scale < 10) {
    printf("scale %u is too small\n", scale);
    return 0;
  }
  if (numT < 2 || numT > omp_get_max_threads())
    numT = omp_get_max_threads();
  omp_set_num_threads(numT);

  init();
  printf("scale %u, n %u, m %u, seed %lu\n", scale, n, m, seed);

  gettimeofday(&start, NULL);
  generate();
  gettimeofday(&stop, NULL);
  sec = (stop.tv_sec - start.tv_sec) +
    (stop.tv_usec - start.tv_usec) / (float)MILLION;
  printf("time for generating the edge list: %.6f sec\n", sec);

  gettimeofday(&start, NULL);
  preprocessing();
  gettimeofday(&stop, NULL);
  sec = (stop.tv_sec - start.tv_sec) +
    (stop.tv_usec - start.tv_usec) / (float)MILLION;
  printf("time for preprocessing: %.6f sec\n", sec);
  kernel1 = sec;

  //run BFS from 64 randomly selected roots
  for (i = 0; i < 64; i++) {
    root = roots[i];
    reset();
    gettimeofday(&start, NULL);
    bfs();
    gettimeofday(&stop, NULL);

    sec = (stop.tv_sec - start.tv_sec) +
      (stop.tv_usec - start.tv_usec) / (float)MILLION;
    minimum = sec;
    cntM = 0;
    for (j = 0; j < n; j++)
      if (parent[j] != -1) {
	cntM += idxCSR[j + 1] - idxCSR[j];
      }
    cm[i] = cntM / 2; //each edge is counted twice
    for (j = 0; j < 30; j++) {
      reset();
      gettimeofday(&start, NULL);
      bfs();
      gettimeofday(&stop, NULL);
      sec = (stop.tv_sec - start.tv_sec) +
	(stop.tv_usec - start.tv_usec) / (float)MILLION;
      minimum = min(minimum, sec);
    }
    runTime[i] = minimum;
    teps[i] = cm[i] / runTime[i];
    if (verbose)
      printf("%9u cm %.0f %.6f %.2f\n", root, cm[i], runTime[i],
	     cm[i] / (runTime[i] * 1000000000));
  }
  printf("\n");
  output(kernel1);
  printf("\n");

  return 0;
}
