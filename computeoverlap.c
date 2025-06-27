#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FNORM   (2.3283064365e-10)
#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define FRANDOM (FNORM * RANDOM)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)
#define sign(x) ((x) > 0 ? 1 : -1)
#define max(a,b) ((a) > (b) ? (a) : (b))
#define N 10000
#define Q 5

struct vector {
  int n[Q];
} zero, sum, group[Q], *perm;

int NoverQ, M, *graph, **neigh, *deg, *color, numUnsat, fact[Q+1];

/* variabili globali per il generatore random */
unsigned myrand, ira[256];
unsigned char ip, ip1, ip2, ip3;

unsigned randForInit(void) {
  unsigned long long y;
  
  y = myrand * 16807LL;
  myrand = (y & 0x7fffffff) + (y >> 31);
  if (myrand & 0x80000000) {
    myrand = (myrand & 0x7fffffff) + 1;
  }
  return myrand;
}

void initRandom(void) {
  int i;
  
  ip = 128;    
  ip1 = ip - 24;    
  ip2 = ip - 55;    
  ip3 = ip - 61;
  
  for (i = ip3; i < ip; i++) {
    ira[i] = randForInit();
  }
}

float gaussRan(void) {
  static int iset = 0;
  static float gset;
  float fac, rsq, v1, v2;
  
  if (iset == 0) {
    do {
      v1 = 2.0 * FRANDOM - 1.0;
      v2 = 2.0 * FRANDOM - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = 1;
    return v2 * fac;
  } else {
    iset = 0;
    return gset;
  }
}

void error(char *string) {
  fprintf(stderr, "ERROR: %s\n", string);
  exit(EXIT_FAILURE);
}

void allocateMem(void) {
  int i;

  fact[0] = 1;
  for ( i = 0; i < Q; i++) {
    zero.n[i] = 0;
    fact[i+1] = (i+1) * fact[i];
  }
  graph = (int*)calloc(2*M, sizeof(int));
  deg = (int*)calloc(N, sizeof(int));
  neigh = (int**)calloc(N, sizeof(int*));
  color = (int*)calloc(N, sizeof(int));
  perm = (struct vector*)calloc(fact[Q], sizeof(struct vector));
}

void printPerm(void) {
  int i, j;
  
  for (i = 0; i < fact[Q]; i++) {
    for (j = 0; j < Q; j++)
      printf("%i", perm[i].n[j]);
    printf("\n");
  }
  printf("\n");
}

struct vector rotate(struct vector input, int modulo) {
  struct vector output;
  int i;

  for (i = 0; i < modulo; i++)
    output.n[i] = (input.n[i] + 1) % modulo;
  return output;
}

void initPerm(int max) {
  int i;
  
  if (max == 1)
    perm[0].n[0] = 0;
  else {
    initPerm(max-1);
    for (i = 0; i < fact[max-1]; i++)
      perm[i].n[max-1] = max-1;
    for (; i < fact[max]; i++)
      perm[i] = rotate(perm[i-fact[max-1]], max);
  }
}
 
void initColors(void) {
  int i;
  
  for (i = 0; i < N; i++)
    color[i] = (int)(FRANDOM * Q);
}
 
double overlapPlanted(void) {
  int i, j, overlap, maxOver=0;

  for (i = 0; i < Q; i++)
    group[i] = zero;
  for (i = 0; i < N; i++)
    group[(int)(i/NoverQ)].n[color[i]]++;
  for (i = 0; i < fact[Q]; i++) {
    overlap = 0;
    for (j = 0; j < Q; j++)
      overlap += group[j].n[perm[i].n[j]];
    if (overlap > maxOver) maxOver = overlap;
  }
  return (double)(Q*maxOver-N)/(Q-1)/N;
}

void freeMem(void) {
  int i;
  for (i = 0; i < N; i++)
    free(neigh[i]);
}


double computeOverlap(int color_assigned[N]){
    allocateMem();
    int i,j, overlap, maxOver=0;
    if (Q * (int)(N/Q) != N) error("Q must divide N");
  NoverQ = (int)(N/Q);
  //M = (int)(0.5 * c * N + 0.5);
  for (i = 0; i < N; i++)
    color[i] = color_assigned[i];
  initRandom();
  initPerm(Q);
  for (i = 0; i < Q; i++)
    group[i] = zero;
  for (i = 0; i < N; i++)
    group[(int)(i/NoverQ)].n[color[i]]++;
  for (i = 0; i < fact[Q]; i++) {
    overlap = 0;
    for (j = 0; j < Q; j++)
      overlap += group[j].n[perm[i].n[j]];
    if (overlap > maxOver) maxOver = overlap;
  }
  double overlap_value = (double)(Q*maxOver-N)/(Q-1)/N;
  return (double)(Q*maxOver-N)/(Q-1)/N;

}
