__kernel void kmeans(
				__global int* data,
				__global int* centroids,
				__const int num_data 
) {

	int id = get_global_id(0);
	
	__local float d_c0, d_c1, d_c2;
	__local float2 pt;
	float2 ctr0,ctr1,ctr2;

	pt.x = data[(id * 3) + 0];
	pt.y = data[(id * 3) + 1];
	
	ctr0.x = centroids[(0 * 2) + 0];
	ctr0.y = centroids[(0 * 2) + 1];
	
	ctr1.x = centroids[(1 * 2) + 0];
	ctr1.y = centroids[(1 * 2) + 1];
	
	ctr2.x = centroids[(2 * 2) + 0];
	ctr2.y = centroids[(2 * 2) + 1];

	d_c0 = distance(pt,ctr0);
	d_c1 = distance(pt,ctr1);
	d_c2 = distance(pt,ctr2);

	if ((int)d_c0 < (int)d_c1 && (int)d_c0 < (int)d_c2)
		data[(3 * id) + 2] = 0;
	else if ((int)d_c1 < (int)d_c0 && (int)d_c1 < (int)d_c2)
		data[(3 * id) + 2] = 1;
	else if ((int)d_c2 < (int)d_c0 && (int)d_c2 < (int)d_c1)
		data[(3 * id) + 2] = 2;

}