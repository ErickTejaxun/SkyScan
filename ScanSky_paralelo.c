/*
* Contar cuerpos celestes
*
* Asignatura Computación Paralela (Grado Ingeniería Informática)
* Código secuencial base
*
* @author Ana Moretón Fernández
* @version v1.2
*
* (c) 2017, Grupo Trasgo, Universidad de Valladolid
*/
/*
Modificiación a versión paralela a través de CUDA
por Erick Tejaxún Xicón 201213050. 
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "cputils.h"


/* Substituir min por el operador */
#define min(x,y)    ((x) < (y)? (x) : (y))

/**
* Funcion secuencial para la busqueda de mi bloque 
*/
int computation(int x, int y, int columns, int* matrixData, int *matrixResult, int *matrixResultCopy){
	// Inicialmente cojo mi indice
	int result=matrixResultCopy[x*columns+y];
	if( result!= -1){
		//Si es de mi mismo grupo, entonces actualizo
		if(matrixData[(x-1)*columns+y] == matrixData[x*columns+y])
		{
			result = min (result, matrixResultCopy[(x-1)*columns+y]);
		}
		if(matrixData[(x+1)*columns+y] == matrixData[x*columns+y])
		{
			result = min (result, matrixResultCopy[(x+1)*columns+y]);
		}
		if(matrixData[x*columns+y-1] == matrixData[x*columns+y])
		{
			result = min (result, matrixResultCopy[x*columns+y-1]);
		}
		if(matrixData[x*columns+y+1] == matrixData[x*columns+y])
		{
			result = min (result, matrixResultCopy[x*columns+y+1]);
		}

		// Si el indice no ha cambiado retorna 0
		if(matrixResult[x*columns+y] == result){ return 0; }
		// Si el indice cambia, actualizo matrix de resultados con el indice adecuado y retorno 1
		else { matrixResult[x*columns+y]=result; return 1;}
		
	}
	return 0; 
}

/**
* Funcion principal
*/
int main (int argc, char* argv[])
{

	/* 1. Leer argumento y declaraciones */
	if (argc < 2) 	{ 		
		printf("Uso: %s <imagen_a_procesar>\n", argv[0]); 		
		return(EXIT_SUCCESS); 	
	} 	
	char* image_filename = argv[1]; 	

	int rows=-1;
	int columns =-1; 
	int *matrixData=NULL; 
	int *matrixResult=NULL; 
	int *matrixResultCopy=NULL; 
	int numBlocks=-1;



	/* 2. Leer Fichero de entrada e inicializar datos */

	/* 2.1 Abrir fichero */
	FILE *f = cp_abrir_fichero(image_filename);

	// Compruebo que no ha habido errores
	if (f==NULL)
	{
	   perror ("Error al abrir fichero.txt");
	   return -1;
	}

	/* 2.2 Leo valores del fichero */
	int i,j,valor;
	fscanf (f, "%d\n", &rows);
	fscanf (f, "%d\n", &columns);
	// Añado dos filas y dos columnas mas para los bordes
	rows=rows+2;
	columns = columns+2; 

	/* 2.3 Reservo la memoria necesaria para la matriz de datos */
	matrixData= (int *)malloc( rows*(columns) * sizeof(int) );
	if ( (matrixData == NULL)   ) {
 		perror ("Error reservando memoria");
	   	return -1;
	}

	/* 2.4 Inicializo matrices */
	for(i=0;i< rows; i++){
		for(j=0;j< columns; j++){
			matrixData[i*(columns)+j]=-1;
		}
	}
	/* 2.5 Relleno bordes de la matriz */
	for(i=1;i<rows-1;i++){
		matrixData[i*(columns)+0]=0;
		matrixData[i*(columns)+columns-1]=0;
	}
	for(i=1;i<columns-1;i++){
		matrixData[0*(columns)+i]=0;
		matrixData[(columns-1)*(columns)+i]=0;
	}
	/* 2.6 Relleno la matriz con los datos del fichero */
	for(i=1;i<rows-1;i++){
		for(j=1;j<columns-1;j++){
			fscanf (f, "%d\n", &matrixData[i*(columns)+j]);
		}
	}
	fclose(f);

	#ifdef WRITE
		printf("Inicializacion \n");
		for(i=0;i<rows;i++){
			for(j=0;j<columns;j++){
				printf ("%d\t", matrixData[i*(columns)+j]);
			}
			printf("\n");
		}
	#endif


	/* PUNTO DE INICIO MEDIDA DE TIEMPO */
	double t_ini = cp_Wtime();

//
// EL CODIGO A PARALELIZAR COMIENZA AQUI
//

     int ndevices = 0;
     cudaError_t cuda_estado = cudaGetDeviceCount(&ndevices);
     if(cuda_status != cudaSuccess)// Si hay error al comunicarnos con la gpu, abortamos.
     {
          fprintf(stderr, "No se ha podido encontrar la GPU NVidia Cuda, estado = %d: %s\n", cuda_status, cudaGetErrorString(cuda_status));
          return cuda_status;
     }

     // Si el dispositivo CUDA está disponible:
     printf("Dispositivo CUDA %d encontrado", ndevices);
     if(!ndevices) return 0;

     //Inicializar el dispotivo CUDA.
	for (int idevice = 0; idevice < ndevices; idevice++)
	{
		config_t* config = configs + idevice;

		// TODO: Configurar dispositvo CUDA  a idevice.
		cudaError_t cuda_estado = cudaSetDevice(idevice);
		if (cuda_status != cudaSuccess) {
		  fprintf(stderr, "no se puede obtener el número del dispositivo CUDA, estado = %d: %s\n",
			cuda_status, cudaGetErrorString(cuda_status));
		  return cuda_status;
		}
		
          // Creando el arreglo de dispositivos para la data de entada
		cuda_estado = cudaMalloc((void**)&config->in_dev, size);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "No se puede localizar el buffer de entada en el dispositivo CUDA %d, estado = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}
		cuda_estado = cudaMalloc((void**)&config->out_dev, size);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "No se puede localizar el buffer de entada en el dispositivo CUDA %d, estado = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}
			
          // Copiar la data de entrada al buffer del dispotivo
		cuda_estado = cudaMemcpy(config->in_dev, input, size,
			cudaMemcpyHostToDevice);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "No se ha podido copiar el buffer de entrada al dispositivo CUDA %d, estado = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}
	       
		printf("Device %d initialized\n", idevice);
	}

     // Comenzamos la ejecución del kerne, cada kernel es ejecutado en cada dispositivo de manera paralela.	
	for (int idevice = 0; idevice < ndevices; idevice++)
	{
		config_t* config = configs + idevice;

		// TODO: Configurar dispositvo CUDA  a idevice.
		cudaError_t cuda_estado = cudaSetDevice(idevice);

		get_time(&config->start);
				
          //Lanzar la preuba del kernel al dispositvio actual
		int estado = pattern2d_gpu(1, nx, 1, 1, ny, 1,
			config->in_dev, config->out_dev, idevice);
		if (status)
		{
			fprintf(stderr, "No se puede ejecutar la parte 2d en el dispotivo %d, estado = %d: %s\n",
				idevice, status, cudaGetErrorString(status));
			return status;
		}
	}
		
     // Sincronizamos la ejecución de los kernels.
	for (int idevice = 0; idevice < ndevices; idevice++)
	{
		config_t* config = configs + idevice;

		// TODO: Configurar dispositvo CUDA  a idevice.
		cudaError_t cuda_estado = cudaSetDevice(idevice);

          /*Esperamos que el dispositivo actual termine de procesar el kernel*/
		cuda_estado = cudaThreadSynchronize();
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "No se pudo sincronziar el hilo del dispositivo %d, estado = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}

		get_time(&finish);

		printf("GPU %d time = %f sec\n", idevice,
			get_time_diff(&config->start, &finish));
	}
	
     // Verificamos los resultados
	for (int idevice = 0; idevice < ndevices; idevice++)
	{
		config_t* config = configs + idevice;

		// TODO: Configurar dispositvo CUDA  a idevice.
			
          // Descargamos los resultados devueltos por la memoria del host.
		cuda_estado = cudaMemcpy(input, config->out_dev, size,
			cudaMemcpyDeviceToHost);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "No se puede copiar los datos de salida del buffer CUDA en el dispositivo %d, estado = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}
		
          // Liberamos el arreglo del dispositivo
		cuda_estado = cudaFree(config->in_dev);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "No se ha podido liberar el buffer de entrada en el dispostivio %d, estado = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}
		cuda_estado = cudaFree(config->out_dev);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "No se ha podido liberar el buffer de entrada en el dispostivio %d, estado = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}

		printf("Device %d deinitialized\n", idevice);

          // Se compara cada reslutado a la CPU
		int maxi = 0, maxj = 0;
		float maxdiff = fabs(input[0] - output[0]);
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				float diff = fabs(
					input[i + j * nx] - output[i + j * nx]);
				if (diff > maxdiff)
				{
					maxdiff = diff;
					maxi = i; maxj = j;
				}
			}
		}

		printf("Device %d result abs max diff = %f @ (%d,%d)\n",
			idevice, maxdiff, maxi, maxj);
	}     

	/* 3. Etiquetado inicial */
	matrixResult= (int *)malloc( (rows)*(columns) * sizeof(int) );
	matrixResultCopy= (int *)malloc( (rows)*(columns) * sizeof(int) );
	if ( (matrixResult == NULL)  || (matrixResultCopy == NULL)  ) {
 		perror ("Error reservando memoria");
	   	return -1;
	}
	for(i=0;i< rows; i++){
		for(j=0;j< columns; j++){
			matrixResult[i*(columns)+j]=-1;
			// Si es 0 se trata del fondo y no lo computamos
			if(matrixData[i*(columns)+j]!=0){
				matrixResult[i*(columns)+j]=i*(columns)+j;
			}
		}
	}



	/* 4. Computacion */
	int t=0;
	/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
	int flagCambio=1; 

	/* 4.2 Busqueda de los bloques similiares */
	for(t=0; flagCambio !=0; t++){
		flagCambio=0; 

		/* 4.2.1 Actualizacion copia */
		for(i=1;i<rows-1;i++){
			for(j=1;j<columns-1;j++){
				if(matrixResult[i*(columns)+j]!=-1){
					matrixResultCopy[i*(columns)+j]=matrixResult[i*(columns)+j];
				}
			}
		}

		/* 4.2.2 Computo y detecto si ha habido cambios */
		for(i=1;i<rows-1;i++){
			for(j=1;j<columns-1;j++){
				flagCambio= flagCambio+ computation(i,j,columns, matrixData, matrixResult, matrixResultCopy);
			}
		}

		#ifdef DEBUG
			printf("\nResultados iter %d: \n", t);
			for(i=0;i<rows;i++){
				for(j=0;j<columns;j++){
					printf ("%d\t", matrixResult[i*columns+j]);
				}
				printf("\n");
			}
		#endif

	}

	/* 4.3 Inicio cuenta del numero de bloques */
	numBlocks=0;
	for(i=1;i<rows-1;i++){
		for(j=1;j<columns-1;j++){
			if(matrixResult[i*columns+j] == i*columns+j) numBlocks++; 
		}
	}

//
// EL CODIGO A PARALELIZAR TERMINA AQUI
//

	/* PUNTO DE FINAL DE MEDIDA DE TIEMPO */
 	double t_fin = cp_Wtime();



	/* 5. Comprobación de resultados */
  	double t_total = (double)(t_fin - t_ini);

	printf("Result: %d\n", numBlocks);
	printf("Time: %lf\n", t_total);
	#ifdef WRITE
		printf("Resultado: \n");
		for(i=0;i<rows;i++){
			for(j=0;j<columns;j++){
				printf ("%d\t", matrixResult[i*columns+j]);
			}
			printf("\n");
		}
	#endif

	/* 6. Liberacion de memoria */
	free(matrixData); 
	free(matrixResult);
	free(matrixResultCopy); 

}
