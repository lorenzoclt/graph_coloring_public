#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define FNORM   (2.3283064365e-10)
#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define FRANDOM (FNORM * RANDOM)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)
#define sign(x) ((x) > 0 ? 1 : -1)
#define max(a,b) ((a) > (b) ? (a) : (b))
#define N 10000   // Numero di nodi nel grafo
#define Q 5 // Numero di intervalli in cui vengono divisi i nodi
#define NUMGRAPHS 16384 // Numero di grafi da generare
#define c 20.0
struct vector {
  int n[Q];
} zero, sum, group[Q], *perm;

int NoverQ, M, *graph, **neigh, *deg, *color, numUnsat, fact[Q+1],**adj_matrix;

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
  adj_matrix = (int**)calloc(N, sizeof(int*));
  for (i = 0; i < N; i++) {
        adj_matrix[i] = (int*)calloc(N, sizeof(int));
    }
  //allocate memory for adjacency matrix


}

void makeGraph(void) {
    int i, var1, var2,j;

    for (i = 0; i < N; i++){
        deg[i] = 0;
        for (j=0;j<N;j++)
            adj_matrix[i][j]=0;
    }
    for (i = 0; i < M; i++) {
        var1 = (int)(FRANDOM * N);
    
        
         do {
            var2 = (int)(FRANDOM * N);
         } while ((int)(var1/NoverQ) == (int)(var2/NoverQ) || adj_matrix[var1][var2]==1);
        graph[2*i] = var1;
        graph[2*i+1] = var2;
        deg[var1]++;
        deg[var2]++;
        adj_matrix[var1][var2]=1;
        adj_matrix[var2][var1]=1;   
    }
    for (i = 0; i < N; i++) {
        neigh[i] = (int*)calloc(deg[i], sizeof(int));
        deg[i] = 0;
    }
    for (i = 0; i < M; i++) {
        var1 = graph[2*i];
        var2 = graph[2*i+1];
        neigh[var1][deg[var1]++] = var2;
        neigh[var2][deg[var2]++] = var1;
     }
     
}
void freeMem(void) {
  int i;
  for (i = 0; i < N; i++)
    free(neigh[i]);
}
int main() {
    FILE *devran = fopen("/dev/urandom","r");
    fread(&myrand, 4, 1, devran);
    fclose(devran);
    initRandom();
    clock_t start, end;
    double elapsed_time;
    M = (int)(0.5 * c * N + 0.5); // Numero di archi nel grafo
    NoverQ = (int)(N/Q);
    int i;
    FILE *file_neigh;
    allocateMem();
    
    // Apri il file per scrivere
    file_neigh = fopen("neigh_dataset_N10000_c20.txt", "w");
    //#file_deg = fopen("deg_dataset.txt", "w");
    start = clock();
    if (file_neigh == NULL) {
        printf("Error: could not open file for writing.\n");
        return 1;
    }
    // genera NUMGRAPHS grafi e scrivi neighbor su file
    for (i = 0; i < NUMGRAPHS; i++) {
        makeGraph();
      
        //every 1000 graphs print the graph number
        if (i % 10 == 0) {
          end = clock(); // Tempo attuale
          elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC; // Tempo in secondi
          printf("Graph number: %d | Time elapsed: %.2f seconds\n", i, elapsed_time);
            
        }

        
        for (int j = 0; j < N; j++) {
             for (int k = 0; k < deg[j]; k++) {
                fprintf(file_neigh, "%d ", neigh[j][k]);
            }
            fprintf(file_neigh, "\n");
        }
        freeMem();
        fprintf(file_neigh, "\n");
        fprintf(file_neigh, "\n");
        fprintf(file_neigh, "\n");
    }

    // Chiudi il file dopo aver scritto tutti i grafi
    fclose(file_neigh);

    
    return 0;
}
