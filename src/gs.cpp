#include <stdio.h>
#include <vector>
#include <iostream>
using namespace std;
int main()
{
   int all_conf[10][5]={{1,-1,1,-1,1},{1,1,-1,1,-1},{-1,1,1,-1,1},{-1,1,-1,1,1},{1,-1,1,1,-1},{-1,1,-1,1,-1},{-1,-1,1,-1,1},{1,-1,-1,1,-1},{1,-1,1,-1,-1},{-1,1,-1,-1,1}} ;
   
  
  int conf[9][5]={0};
  int num=0;
  
 for (int a=0;a<10;a++){
	 for(int j=0;j<5;j++){
		 conf[0][j]=all_conf[a][j];
	 }
	 for(int b=0;b<10;b++){
		 for(int k=0;k<5;k++){
			 conf[1][k]=all_conf[b][k];
		 }
		 for(int c =0;c<10;c++){
			 for(int l=0;l<5;l++){
				conf[2][l]=all_conf[c][l];
			 }
			 for(int d =0;d<10;d++){
				 for(int m=0;m<5;m++){
					 conf[3][m]=all_conf[d][m];
				 }
				 for(int e =0;e<10;e++){
					 for(int n=0;n<5;n++){
						 conf[4][n]=all_conf[e][n];
					 }
					 for (int f =0;f<10;f++){
						 for(int o=0;o<5;o++){
							 conf[5][o]=all_conf[f][o];
						 }
						 for(int g=0;g<10;g++){
							 for(int p=0;p<5;p++){
								 conf[6][p]=all_conf[g][p];
							 }
							 for(int h=0;h<10;h++){
								 for(int q=0;q<5;q++){
									 conf[7][q]=all_conf[h][q];
								 }
								 for(int i=0;i<10;i++){
									 for(int r=0;r<5;r++){
										 conf[8][r]=all_conf[i][r];
									 }

													 if(conf[6][3]!=conf[7][0] &&
                                                        conf[6][2]!=conf[3][3] &&
                                                        conf[3][2]!=conf[7][1]  &&
                                                        conf[2][3]!=conf[7][2]   &&
                                                       conf[2][4]!=conf[3][1]  &&
                                                       conf[3][0]!=conf[0][4] &&
                                                       conf[2][0]!=conf[0][3] &&
                                                       conf[2][2]!=conf[8][1] &&
                                                       conf[0][1]!=conf[1][0]  &&
                                                       conf[1][2]!=conf[5][1]&&
                                                       conf[1][3]!=conf[4][2] &&
                                                       conf[4][3]!=conf[5][0]  &&
                                                       conf[6][0]!=conf[5][3] &&
                                                       conf[4][0]!=conf[8][3] &&
											           conf[7][3]!=conf[8][0]&&
													   
													  conf[3][4]*conf[0][0]+conf[0][0]*conf[1][1]+conf[1][1]*conf[5][2]+conf[5][2]*conf[6][1]+conf[6][1]*conf[3][4]==-3&&
													  conf[1][4]*conf[4][1]+conf[4][1]*conf[8][2]+conf[8][2]*conf[2][1]+conf[2][1]*conf[0][2]+conf[0][2]*conf[1][4]==-3&&
													  conf[7][4]*conf[6][4]+conf[6][4]*conf[5][4]+conf[5][4]*conf[4][4]+conf[4][4]*conf[8][4]+conf[8][4]*conf[7][4]==-3
													  
													   
													   )
											            num+=1;					 
								 }
							 }
						 }
					 }
					 
					 
				 }
			 }
		 }
	 } 
	 
 }
  
cout<<num<<endl;

}

  