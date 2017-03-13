#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <omp.h>

#define MILLION 1000000L
#define DENSE   0
#define SPARSE  1
#define WORD    6          //64 bits, 2^6
#define MASK    0x0000003F //6 1's

unsigned verbose = 0;
unsigned NumT;             //number of threads
unsigned *prefixSum;       //thread offsets of where to write in sparse

unsigned n, m, bitN;       //num vertices, num edges, bitN is n >> WORD
unsigned root = 0;         //root of BFS, default is vertex 0
unsigned *row, *rowOffset; //compressed sparse row
int *parent;

unsigned layerSize, layerOutDeg;
//threshold to switch from sparse to dense: n/2^5 or n/32
//dense representation of layers
uint64_t *dense, *newDense;
//sparse representation of layers
unsigned *sparse, sparseSize, *newSparse, newSparseSize;
unsigned **mySparse, *mySparseSize;

//lookup tables
uint64_t oneBit[64];
//oneBit[0] 0001
//oneBit[1] 0010
//oneBit[2] 0100
//oneBit[3] 1000

static unsigned s2u(const char s[]){
  unsigned a = 0;

  while (*s != '\n')
    a = a*10 + (*(s++) - '0');
  return a;
}

void readAdjGraph(char *filename){
  uint64_t i, *off;
  unsigned numItem;
  struct stat buf;
  char *buffer;
  int status;
  FILE *fp;

  status = stat(filename, &buf);
  if (status){
    printf("no such file: %s\n", filename);
    exit(0);
  }
  printf("%s: %lu bytes\n", filename, buf.st_size);

  buffer = (char*)malloc(buf.st_size);
  off = (uint64_t*)malloc(sizeof(uint64_t) * buf.st_size);

  fp = fopen(filename, "rb");
  if (!fp){
    printf("can't open file: %s\n", filename);
    exit(0);
  }
  fread((void*)buffer, 1, buf.st_size, fp);
  fclose(fp);

  for (i=0, numItem=0; i<buf.st_size; i++)
    if (buffer[i] == '\n')
      off[numItem++] = i+1;

  buffer[off[0]-1] = 0;
  if (strcmp(buffer, "AdjacencyGraph") != 0){
    printf("file format is not AdjacencyGraph: %s\n", filename);
    exit(0);
  }

  n = s2u(buffer + off[0]);
  posix_memalign((void **)&rowOffset, 64, sizeof(unsigned) * (n+1));
  m = s2u(buffer + off[1]);
  posix_memalign((void **)&row,       64, sizeof(unsigned) * m);
  printf("n %u, m %u\n", n, m);

#pragma omp parallel for default(shared)
  for (i=0; i<n; i++)
    rowOffset[i] = s2u(buffer + off[i+2]);

  rowOffset[n] = m;

#pragma omp parallel for default(shared)
  for (i=0; i<m; i++)
    row[i] = s2u(buffer + off[i+n+2]);

  free(off);
  free(buffer);
}

void reset(void){
  unsigned i, o, w;

  for (i=0; i<n; i++)
    parent[i] = -1;
  parent[root] = root;
  for (i=0; i<bitN; i++)
    newDense[i] = 0;

  //initialize the first layer
  newSparse[0] = root;
  layerSize = newSparseSize = 1;
  layerOutDeg = rowOffset[root+1] - rowOffset[root];

  w = root >> WORD;  //64-bit word number
  o = (root & MASK); //offset within word
  newDense[w] = oneBit[o];
}

void init(char *filename){
  unsigned i;

  readAdjGraph(filename);
  posix_memalign((void **)  &parent,      64, sizeof(int) * n);
  posix_memalign((void **)   &dense,      64, n >> 3); // n/8 bytes
  posix_memalign((void **)&newDense,      64, n >> 3); // n/8 bytes
  bitN = n >> WORD;
  posix_memalign((void **)   &sparse,     64, sizeof(unsigned) * (n>>5));
  posix_memalign((void **)&newSparse,     64, sizeof(unsigned) * (n>>5));
  posix_memalign((void **)&mySparse,      64, sizeof(unsigned*) * NumT);
  for (i=0; i<NumT; i++)
    posix_memalign((void **)&mySparse[i], 64, sizeof(unsigned) * (n>>5));
  posix_memalign((void **)&mySparseSize,  64, sizeof(unsigned) * NumT);
  posix_memalign((void **)&prefixSum,     64, sizeof(unsigned) * (NumT+1));
  oneBit[0] = 1;
  for (i=1; i<64; i++)
    oneBit[i] = oneBit[i-1] << 1;
  reset();
}

unsigned sparseLayer(void){
  unsigned tid, i, j, u, v, outDeg = 0;

#pragma vector aligned
  if (verbose) printf("sparse\n");
  for (i=0; i<NumT; i++) mySparseSize[i] = 0;

  if (sparseSize == 1){
    u = sparse[0];
#pragma omp parallel for default(shared) private(tid,v) \
  reduction(+:outDeg) schedule(guided)
    for (i=rowOffset[u]; i<rowOffset[u+1]; i++){
      tid = omp_get_thread_num();
      v = row[i];
      if (parent[v] == -1){
	parent[v] = u;
	mySparse[tid][mySparseSize[tid]++] = v;
	outDeg += rowOffset[v+1] - rowOffset[v];
      }
    }
  }
  else { //sparseSize > 1
#pragma omp parallel for default(shared) private(tid,j,u,v)	\
  reduction(+:outDeg) schedule(guided)
    for (i=0; i<sparseSize; i++){
      tid = omp_get_thread_num();
      u = sparse[i];
      for (j=rowOffset[u]; j<rowOffset[u+1]; j++){
	v = row[j];
	if (parent[v] == -1){
	  //race condition, multiple copies of v may be added
	  parent[v] = u;
	  mySparse[tid][mySparseSize[tid]++] = v;
	  outDeg += rowOffset[v+1] - rowOffset[v];
	}
      }
    }
  }

  layerOutDeg = outDeg;
  prefixSum[0] = 0;
  for (i=0; i<NumT; i++)
    prefixSum[i+1] = prefixSum[i] + mySparseSize[i];
  newSparseSize = prefixSum[NumT];
#pragma omp parallel for default(shared) private(tid,j)
  for (i=0; i<NumT; i++){
    tid = omp_get_thread_num();
    for (j=0; j<mySparseSize[tid]; j++)
      newSparse[prefixSum[tid] + j] = mySparse[tid][j];
  }
  return newSparseSize;
}

void sparse_2_dense(void){
  unsigned i, j, u, w, o, chunk, rem, start, end;

#pragma vector aligned
  chunk = (n >> 9) / NumT;
  rem   = (n >> 9) % NumT;

#pragma omp parallel for default(shared)
  for (i=0; i<bitN; i++)
    dense[i] = 0;

#pragma omp parallel for default(shared) private(start,end,j,u,w,o)	\
  schedule(static,1)
  for (i=0; i<NumT; i++){
    start = i * chunk;
    end = start + chunk;
    if (rem){
      if (i < rem) start +=   i, end += i + 1;
      else         start += rem, end += rem;
    }
    start <<= 9, end <<= 9;
    for (j=0; j<sparseSize; j++){
      u = sparse[j];
      if (u < start || u >= end)
	continue;
      w = u >> WORD;  //64-bit word number
      o = (u & MASK); //offset within word
      dense[w] = dense[w] | oneBit[o];
    }
  }
}

unsigned dense_2_sparse(void){
  unsigned tid, i, j, k, a, d, e, u, v, numV = 0, outDeg = 0;
  uint32_t *ptr32;
  uint16_t *ptr16;
  uint8_t  *ptr8, mask;

#pragma vector aligned
  if (verbose) printf("dense to sparse\n");
  for (i=0; i<NumT; i++) mySparseSize[i] = 0;

#pragma omp parallel for default(shared)		\
  private(tid,j,k,u,v,a,d,e,ptr32,ptr16,ptr8,mask)	\
  reduction(+:numV,outDeg) schedule(guided)
  for (i=0; i<bitN; i++){
    tid = omp_get_thread_num();
    if (dense[i] == 0) continue; //dense[i] is all zeros
    ptr32 = (uint32_t*) &dense[i];
    for (j=0; j<2; j++){
      if (ptr32[j] == 0) continue;
      ptr16 = (uint16_t*) &ptr32[j];
      for (k=0; k<2; k++){
	if (ptr16[k] == 0) continue;
	ptr8 = (uint8_t*) &ptr16[k];
	for (a=0; a<2; a++){
	  if (ptr8[a] == 0) continue;
	  mask = ptr8[a];
	  d = 0;
	  while (mask){
	    if (mask & 1){
	      u = (i<<6) + (j<<5) + (k<<4) + (a<<3) + d;
	      //visit vertex u
	      for (e=rowOffset[u]; e<rowOffset[u+1]; e++){
		v = row[e];
		if (parent[v] == -1){
		  //race condition, multiple copies of v may be added
		  parent[v] = u;
		  mySparse[tid][mySparseSize[tid]++] = v;
		  outDeg += rowOffset[v+1] - rowOffset[v];
		  numV++;
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

  layerOutDeg = outDeg;
  prefixSum[0] = 0;
  for (i=0; i<NumT; i++)
    prefixSum[i+1] = prefixSum[i] + mySparseSize[i];
#pragma omp parallel for default(shared) private(tid,j)
  for (i=0; i<NumT; i++){
    tid = omp_get_thread_num();
    for (j=0; j<mySparseSize[tid]; j++)
      newSparse[prefixSum[tid] + j] = mySparse[tid][j];
  }
  newSparseSize = numV;
  return newSparseSize;
}

int denseLayer(void){
  unsigned i, j, u, w, o, numV = 0, outDeg = 0;
  uint64_t tmp;

#pragma vector aligned
  if (verbose) printf("dense\n");

#pragma omp parallel for default(shared)
  for (i=0; i<bitN; i++)
    newDense[i] = 0;

#pragma omp parallel for default(shared) private(j,u,w,o,tmp)	\
  reduction(+:numV,outDeg) schedule(dynamic,512)
  for (i=0; i<n; i++){
    if (parent[i] != -1) continue;
    for (j=rowOffset[i]; j<rowOffset[i+1]; j++){
      u = row[j];     //u has an edge going into vertex i
      w = u >> WORD;  //64-bit word number
      o = (u & MASK); //offset within word
      tmp = dense[w] & oneBit[o];
      //Is u in the layer?
      if (tmp == 0) //No.
	continue;
      //Add vertex i to the new layer
      parent[i] = u;
      w = i >> WORD;  //64-bit word number
      o = (i & MASK); //offset within word
      newDense[w] = newDense[w] | oneBit[o];
      outDeg += (rowOffset[i+1] - rowOffset[i]); //i's out-degreees
      numV++;
      break;
    }
  }

  layerOutDeg = outDeg;
  return numV;
}

void bfs(void){
  unsigned numIter = 0, prevIter = SPARSE, *tmpPtr;
  uint64_t *ptr64;

  if (verbose) printf("threshold: %u\n", n>>5);

  while (layerSize){
    if (verbose)
      printf("iter %u, numV %u, outDeg %u, ",numIter, layerSize, layerOutDeg);
    numIter++;
    if (prevIter == SPARSE){
      tmpPtr = sparse, sparse = newSparse, newSparse = tmpPtr;
      sparseSize = newSparseSize;
      if (layerSize > (n>>5) || layerOutDeg > (n>>5)){
	sparse_2_dense();
	layerSize = denseLayer();
	prevIter = DENSE;
      }
      else
	layerSize = sparseLayer();
    }
    else {
      ptr64 = dense, dense = newDense, newDense = ptr64;
      if (layerSize > (n>>5) || layerOutDeg > (n>>5))
	layerSize = denseLayer();
      else {
	layerSize = dense_2_sparse();
	prevIter = SPARSE;
      }
    }
  }

  if (verbose) printf("number of iterations: %u\n", numIter);
}

int main(int argc, char* argv[]){
  int c;
  char *filename;
  unsigned i, cn, cm;
  struct timeval start, stop;
  float sec, sum;

  NumT = omp_get_max_threads();
  filename = NULL;
  while ((c = getopt(argc, argv, "f:r:t:v")) != -1){
    switch (c){
    case 'f':
      filename = (char*)malloc(sizeof(char) * (strlen(optarg) + 1));
      strcpy(filename, optarg);
      break;
    case 'r': sscanf(optarg, "%u", &root); break;
    case 't': sscanf(optarg, "%u", &NumT); break;
    case 'v': verbose = 1; break;
    default: break;
    }
  }
  for (i=optind; i<argc; i++){
    filename = (char*)malloc(sizeof(char) * (strlen(argv[i]) + 1));
    strcpy(filename, argv[i]);
  }
  if (!filename){
    printf("filename?\n");
    return 0;
  }

  if (NumT < 2 || NumT > omp_get_max_threads())
    NumT = omp_get_max_threads();
  init(filename);
  omp_set_num_threads(NumT);

  gettimeofday(&start, NULL);
  bfs();
  gettimeofday(&stop, NULL);
  sec = (stop.tv_sec - start.tv_sec) +
    (stop.tv_usec - start.tv_usec) / (float)MILLION;
  printf("ompBFS, %u threads: %.6f\n", NumT, sec);
  if (verbose){
    cn = cm = 0;
    for (i=0; i<n; i++)
      if (parent[i] != -1){
	cn++;
	cm += rowOffset[i+1] - rowOffset[i];
      }
    printf("n-prime %u, m-prime %u\n", cn, cm);
    verbose = 0;
  }
  printf("\n");

  sum = 0;
  for (i=0; i<3; i++){
    reset();
    gettimeofday(&start, NULL);
    bfs();
    gettimeofday(&stop, NULL);
    sec = (stop.tv_sec - start.tv_sec) +
      (stop.tv_usec - start.tv_usec) / (float)MILLION;
    printf("ompBFS, %u threads: %.6f\n", NumT, sec);
    sum += sec;
  }
  printf("\naverage %.6f\n", sum/3);

  return 0;
}
